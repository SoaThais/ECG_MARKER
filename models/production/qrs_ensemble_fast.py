"""PATCH -- NOT part of the vendored production dump (qrs_ensemble.py /
qrs_model.py). Those two ship exactly as pulled from daint and are marked
"do not edit" in their own README; this file is our own addition on top,
kept separate for that reason.

qrs_ensemble.QRSEnsemble.predict() runs its K members in a plain Python loop:

    for h in self.heads:
        out = h(window, emb, routing_alpha=routing_alpha)
        cols.append(out[2].float().cpu().numpy())   # GPU->CPU sync every iteration

That's K sequential forward passes plus K GPU->CPU syncs, when every member
is architecturally identical (same MaskHeadV6 model_kwargs, only the weights
differ) and consumes the exact same (window, emb) input. This module stacks
all K members' parameters/buffers with torch.func.stack_module_state and
runs them as a single vmap'd forward pass instead -- one kernel-launch
sequence instead of K, one GPU->CPU sync instead of K.

Usage
-----
    from qrs_ensemble import QRSEnsemble
    from qrs_ensemble_fast import FastQRSEnsemble

    ens = QRSEnsemble.load("megamodel_loo32.pt", device="cuda")
    fast = FastQRSEnsemble.from_ensemble(ens)
    out = fast.predict(window, emb)   # identical dict contract to ens.predict()

FastQRSEnsemble.predict()'s output is numerically verified against both
QRSEnsemble.predict() (see self-test below) and the shipped golden fixtures
-- run this file directly to check both on whatever bundle files are present
alongside it.

Why BatchNorm needed inlining
------------------------------
cudnn's BatchNorm kernel does not support vmap's batching rule at all (hard
error: "NYI: querying is_contiguous inside of vmap"). MaskHeadV6 uses
BatchNorm2d (in `compress`) and BatchNorm1d (at the front of the tau heads).
The naive fix -- torch.backends.cudnn.flags(enabled=False) around the vmapped
call -- makes it run, but disables cudnn's fast Conv kernels for that whole
forward pass too, not just BatchNorm; measured 4.5x SLOWER than the original
loop on a GTX 1080 (32 members, batch=16: 497ms loop vs 2264ms that way).

Instead, _FunctionalBatchNorm below replaces every BatchNorm1d/2d submodule
on the (deep-copied, meta-device) template with a module that computes the
identical eval-mode affine transform -- (x - running_mean) / sqrt(running_var
+ eps) * weight + bias -- as plain tensor ops, never calling into cudnn at
all. Same parameter/buffer names as the originals, so stack_module_state()
(run on the REAL heads, never touched) picks up the real running stats/
weights unchanged; only the template's graph differs. Conv2d stays on cudnn
throughout since it isn't touched. Only valid for eval mode (this bundle's
heads are always eval()) -- BatchNorm's train-mode batch statistics are not
replicated here.

With that fix, cudnn stays fully enabled and vmap no longer crashes or
regresses -- but on our GTX 1080 it also isn't a real win: measured 304ms
(loop) vs 317ms (this module) per batch of 16, 32 members -- 0.96x, a wash
within run-to-run noise, not a speedup. Numerics are much tighter than the
disabled-cudnn attempt too (max diff ~0.006ms vs golden-level ~0.03ms
before), since both paths now run identical cudnn kernels.

Cross-hardware check, RTX 4070 Ti SUPER (Ada Lovelace, MMC workstation
node114) -- same torch 2.6.0+cu124, batch=32, 32 members: loop=118.5ms,
this module=227.2ms -- 0.52x, i.e. nearly 2x SLOWER, WORSE than on the
GTX 1080, not better. So "maybe it helps on a bigger/newer GPU" was tested
directly and the answer is no: as the underlying loop gets faster on
better hardware, vmap's own batching-rule dispatch overhead (roughly
fixed regardless of GPU) becomes a LARGER fraction of the now-smaller
total time, not a smaller one. This isn't a Pascal-specific or
small-model-specific artifact -- it's a real cost of this approach.

'light' mode (models/production's 4-member subset, see ecg_nn.recording)
remains the effective speedup: a genuine ~7.4x from doing less work (4
members instead of 32), not from parallelizing the same work differently.
"""
import copy
import os

import numpy as np
import torch
import torch.nn as nn
from torch.func import stack_module_state, functional_call


class _FunctionalBatchNorm(nn.Module):
    """Eval-mode-only BatchNorm1d/2d replacement using plain tensor ops
    instead of cudnn's batch_norm kernel -- see module docstring for why.
    Same parameter/buffer names as nn.BatchNorm{1,2}d (weight, bias,
    running_mean, running_var), so stack_module_state() on the ORIGINAL
    heads still finds them; this class only ever appears in the meta-device
    template functional_call runs against, never in a real head.
    """
    def __init__(self, num_features, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features))
        self.bias = nn.Parameter(torch.empty(num_features))
        self.register_buffer('running_mean', torch.empty(num_features))
        self.register_buffer('running_var', torch.empty(num_features))
        self.eps = eps

    def forward(self, x):
        # x: (N, C) for BatchNorm1d or (N, C, H, W) for BatchNorm2d --
        # broadcast every per-channel stat/param over all dims but channel.
        shape = [1, -1] + [1] * (x.dim() - 2)
        mean = self.running_mean.view(*shape)
        var  = self.running_var.view(*shape)
        w    = self.weight.view(*shape)
        b    = self.bias.view(*shape)
        return (x - mean) / torch.sqrt(var + self.eps) * w + b


def _inline_batchnorm(module):
    """Recursively replace every BatchNorm1d/2d submodule in-place with
    _FunctionalBatchNorm (same num_features/eps, fresh -- never-initialized
    -- parameters; the real values come from stack_module_state on the
    original heads at call time, this copy's own values are never read).
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            setattr(module, name, _FunctionalBatchNorm(child.num_features, child.eps))
        else:
            _inline_batchnorm(child)
    return module


class FastQRSEnsemble:
    """vmap-parallel drop-in for qrs_ensemble.QRSEnsemble's predict() path.

    Only predict() is provided -- load a qrs_ensemble.QRSEnsemble the normal
    way first (that part isn't the bottleneck) and wrap it with
    from_ensemble(). check_encoder() etc. are still on the wrapped instance.
    """

    def __init__(self, heads, provenance, manifest, device):
        self.provenance = list(provenance)
        self.manifest = dict(manifest)
        self.device = torch.device(device)

        # A single member as a stateless "meta" template -- functional_call
        # runs this shape/graph with whichever (stacked) params/buffers are
        # handed to it per vmap slice, so the template's own weights are
        # never read. BatchNorm inlined here (see module docstring); Conv2d
        # etc. untouched, still dispatch to cudnn normally.
        template = _inline_batchnorm(copy.deepcopy(heads[0]))
        self._template = template.to('meta')
        self._params, self._buffers = stack_module_state(heads)   # each: {name: (K, *shape)}

        def _call_one(params, buffers, window, emb, routing_alpha):
            return functional_call(self._template, (params, buffers),
                                    (window, emb), {'routing_alpha': routing_alpha})

        # in_dims: params/buffers batched over their leading (member) dim;
        # window/emb/routing_alpha broadcast unchanged to every member.
        self._vmapped = torch.vmap(_call_one, in_dims=(0, 0, None, None, None))

    @classmethod
    def from_ensemble(cls, ensemble):
        """Wrap an already-loaded qrs_ensemble.QRSEnsemble."""
        return cls(ensemble.heads, ensemble.provenance, ensemble.manifest, ensemble.device)

    def __len__(self):
        return len(self.provenance)

    @torch.no_grad()
    def predict(self, window, emb, routing_alpha=1.0, return_members=False):
        """Same contract as QRSEnsemble.predict() -- see that docstring."""
        window = torch.as_tensor(window).to(self.device)
        emb = torch.as_tensor(emb).to(self.device)

        out = self._vmapped(self._params, self._buffers, window, emb, routing_alpha)
        coefs_b = out[2]                                  # (K, N, 4)
        C = coefs_b.float().cpu().numpy()

        on = np.median(C[:, :, 0], axis=0)
        off = np.median(C[:, :, 2], axis=0)
        res = {
            "onset": on,
            "offset": off,
            "duration": off - on,
            "tau_on": np.median(C[:, :, 1], axis=0),
            "tau_off": np.median(C[:, :, 3], axis=0),
            "spread_on": C[:, :, 0].std(axis=0, ddof=1) if len(self) > 1 else np.zeros_like(on),
            "spread_off": C[:, :, 2].std(axis=0, ddof=1) if len(self) > 1 else np.zeros_like(off),
            "n_members": len(self),
        }
        if return_members:
            res["members"] = C
        return res


def _self_test():
    """Load whatever bundle(s) are present alongside this file, compare
    FastQRSEnsemble against both the original QRSEnsemble.predict() and the
    shipped golden fixture, on CPU (matching how the goldens were recorded).
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from qrs_ensemble import QRSEnsemble

    here = os.path.dirname(__file__)
    pairs = [
        ('megamodel_loo32.pt', 'golden_megamodel_loo32.npz'),
        ('allseed64.pt', 'golden_allseed64.npz'),
    ]
    ran_any = False
    for bundle, golden in pairs:
        bundle_path, golden_path = os.path.join(here, bundle), os.path.join(here, golden)
        if not (os.path.exists(bundle_path) and os.path.exists(golden_path)):
            continue
        ran_any = True
        print(f"=== {bundle} ===")
        g = np.load(golden_path)
        ens = QRSEnsemble.load(bundle_path, device='cpu')
        fast = FastQRSEnsemble.from_ensemble(ens)

        slow_out = ens.predict(g['window'], g['emb'])
        fast_out = fast.predict(g['window'], g['emb'])

        for k in ('onset', 'offset', 'duration', 'tau_on', 'tau_off'):
            d_vs_slow = np.abs(fast_out[k] - slow_out[k]).max()
            d_vs_golden = np.abs(fast_out[k] - g[k]).max()
            print(f"  {k:10s} fast-vs-loop={d_vs_slow:.6f}  fast-vs-golden={d_vs_golden:.4f}"
                  + ("  (golden > 0.5ms noise floor for this bundle, informational only)"
                     if d_vs_golden >= 0.5 else ""))
            # fast-vs-loop is the real correctness gate: same hardware, same
            # process, so any divergence here is from the BatchNorm-inlining
            # rewrite itself, not cross-run/cross-hardware noise. Same
            # tolerance class as the vendored run_golden() -- tight enough to
            # catch a real bug (README measured a wrong-weights bug moving
            # durations by 144ms) but loose enough for legitimate
            # reduction-order noise between vmap's batched matmuls and the
            # sequential loop's.
            #
            # fast-vs-golden is informational, not asserted: allseed64 (64
            # members) already exceeds golden's 0.5ms tolerance on the
            # STOCK, unmodified loop implementation (0.627ms measured
            # earlier on GPU) -- larger member count means more
            # floating-point accumulation noise, independent of this patch.
            assert d_vs_slow < 0.5, f"{bundle}/{k}: fast diverges from the original loop by {d_vs_slow}"
        print("  OK")

    if not ran_any:
        print("no bundle+golden pairs found next to this file -- nothing to test")


if __name__ == '__main__':
    _self_test()
