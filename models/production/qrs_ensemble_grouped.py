"""PATCH -- NOT part of the vendored production dump (qrs_ensemble.py /
qrs_model.py). Those two ship exactly as pulled from daint and are marked
"do not edit" in their own README; this file is our own addition on top,
kept separate for that reason.

The previous two attempts at this (qrs_ensemble_fast.py's vmap, and the
CUDA-graphs one) both tried to eliminate DISPATCH overhead, on the theory
that many-tiny-kernel-launches was the bottleneck. Measured GPU utilization
turned out to be 0.45% (26 GFLOPs of real work vs 656ms of GPU time on a
GTX 1080 rated for 8.87 TFLOPS) -- 200x off from what raw compute would
need. So the bottleneck isn't dispatch overhead OR FLOP throughput: it's
that hundreds of individually-launched kernels are each too SMALL to
occupy the GPU (low occupancy), and each one still pays fixed per-kernel
scheduling latency at the hardware level -- something neither vmap's
dispatch-side fix nor CUDA graphs' launch-side fix touches, since both
still end up issuing the same number of individually-scheduled kernels to
the GPU, just via a different path.

Isolated proof this is the real lever, not a dead end like the other two:
one conv1d layer, K=32 members, same total work --

    32 separate conv1d calls: 1.016ms
    1 grouped conv1d (groups=32): 0.326ms  -- 3.12x

Manually reshaping K members' distinct weights into ONE grouped-convolution
call (or one batched-matmul call, for the Linear layers) genuinely collapses
K small kernels into 1 bigger one that actually keeps the GPU's execution
units busy -- unlike vmap, which does something conceptually similar
automatically but pays enough of its own dispatch tax to cancel the gain.

This module reimplements MaskHeadV6._forward_impl entirely by hand, batching
the K-member dimension into every operation from the start using that
technique throughout, instead of wrapping/patching the existing per-member
modules the way the other two patches do. Two consequences of doing it by
hand:

  1. It's HARDCODED to this bundle's actual model_kwargs (verified identical
     across both megamodel_loo32.pt and allseed64.pt): tau_arch='linear',
     tau_feats='no_hubert', offset_prior=False, no_compress=False,
     compress_mode='full', center_context=False, use_ste=False,
     feature_pool=False, detach_routing=False, tau_detach=False,
     tau_cap=None, scale=1.0 (C=32), scorer_scale=0.5 (SC=16). Every other
     branch qrs_model.py's MaskHeadV6 supports for other configs (e.g.
     compress_mode='small'/'avgpool', tau_arch='mlp64'/'mlp16', the
     feature_pool / offset_prior / center_context / use_ste / no_compress
     True-branches) is simply not implemented here -- a different bundle
     built with different flags would need different stacking logic, not
     just different weights. This module will raise if handed a bundle
     whose model_kwargs don't match what it was written for.
  2. routing_compressor's output (feat_hub) is computed by the original
     forward pass but never used when tau_feats='no_hubert' (the tau head's
     input only concatenates [feat_lead, hand_feat]) -- so this
     reimplementation skips calling it entirely. Its weights are still
     present in the checkpoint (loaded, just unused), same as the original.
     coefs_a, mask, durations, film_m, and the per-lead attention/backbone
     outputs qrs_model.py also returns are likewise never computed here --
     QRSEnsemble.predict() only ever reads out[2] (coefs_b), so nothing
     else is needed.

Usage
-----
    from qrs_ensemble import QRSEnsemble
    from qrs_ensemble_grouped import GroupedQRSEnsemble

    ens = QRSEnsemble.load("megamodel_loo32.pt", device="cuda")
    grouped = GroupedQRSEnsemble.from_ensemble(ens)
    out = grouped.predict(window, emb)   # identical dict contract to ens.predict()

Result, GTX 1080, batch=32, 32 members: loop=592.2ms, deferred (already
wired in)=566.5ms, this module=538.6ms -- 1.10x over the loop, 1.05x over
deferred, i.e. a real but modest win, not the 3.12x the isolated one-layer
test predicted. Profiling this module's own forward pass explains why:
cudnn_convolution is still the single biggest item (31% of GPU time) even
though every conv is now genuinely grouped -- grouping the SAME layer to
groups=64 or groups=32 (K=32 members x each layer's own groups=1 or 2)
doesn't scale as cleanly as the isolated groups=32 test did, likely a
Pascal-generation/cuDNN limit on wide-group kernels rather than something
fixable in this module. The rest (reflection-pad, GELU, bias-add) is still
one kernel launch per LAYER -- down from one per (layer x member) in the
loop, ~32x fewer calls -- but on this hardware each of those launches still
costs enough that, combined, they roughly match the convolution time.
Correctness: exact-shape match against QRSEnsemble.predict() on both
bundles, max diff ~0.0001ms (pure floating-point reduction-order noise,
same class as vmap's), see _self_test() below.

Beat-batch scaling, GTX 1080 (before the compress fix below): B=8 gave
1.51x over deferred, B=32 gave 1.05x, B=64 collapsed to 0.18x (5.6x
*slower*), B=128 OOM'd outright on this 8GB card. Root cause: `compress`
(the routing/HuBERT branch) is the only stacked layer whose FIRST conv
takes the raw, K-independent 768-channel embedding as input -- it was
being `.repeat(1, K, 1, 1)`'d to 768*K=24576 channels before a
groups=K conv, materializing a ~2.9GB tensor at B=64/K=32 alone. Since
that layer's per-member conv is already ungrouped (groups=1: every
output channel already mixes all 768 input channels), a plain groups=1
conv over the stacked K*C output channels computes the identical result
directly from the UN-repeated 768-channel input -- no group multiplication,
no repeat, and a far more standard cudnn kernel shape. Fixed here; layers
1-4 of compress (and the backbone/scorer, which are genuinely per-member
block-grouped, not just replicated) are unaffected and still need
groups=K.

Re-benchmarked on the MMC workstation (RTX 4070 Ti SUPER, 16GB) after the
compress fix -- two results, both against wiring this in:

    B      deferred    grouped    speedup
    8       52.1ms      39.0ms    1.33x
    32     115.1ms     155.7ms    0.74x  (35% SLOWER -- production's actual batch size)
    64     277.0ms     311.6ms    0.89x
    128    614.2ms     OOM        --
    256   1260.5ms     OOM        --
    512   2511.8ms     OOM        --

1. The compress fix alone wasn't enough -- grouped still OOMs at B=128 on
   a 16GB card. The backbone is the next offender: it still multiplies
   BOTH groups and channel width by K (groups=K*2, width=K*C), so its
   widest activation is (N*L, K*C, W) = (N*12, 1024, 550) at K=32 -- ~3.5GB
   at B=128 by itself, several such tensors alive per forward pass. Fixing
   this the same way compress was fixed isn't possible -- the backbone's
   groups=2 is a real per-member internal structure (not incidental like
   compress's groups=1-per-member-then-repeated), so avoiding the K
   multiplication would need the "group-major" weight reordering discussed
   above, which in turn requires an explicit channel permute before the
   scorer to restore member-contiguous blocks. Not implemented -- judged
   not worth the added correctness risk given finding 2.
2. At B=32 -- the batch size ecg_nn/recording.py actually uses -- grouped
   is 35% SLOWER than the loop on this GPU, a regression from the GTX
   1080's modest 1.10x. This is the same pattern qrs_ensemble_fast.py's
   vmap showed (0.96x on GTX 1080, 0.52x on this same RTX card): as the
   underlying loop gets faster on better hardware, this approach's own
   fixed overhead (input repeats, cuDNN's algorithm selection for
   wide-group convolutions) stops paying for itself and becomes a net
   loss instead of a smaller win.

Net: a third independent technique (after vmap and CUDA graphs) that does
not generalize across hardware/batch-size combinations, with this one
actively regressing on the newer GPU at the real production batch size.
NOT recommended for wiring in -- kept as a documented, unused patch like
the other two. qrs_ensemble_deferred.py remains the one technique that
held up (~4.5x real-world, from batching + deferred sync) across both
GPUs and every batch size tested.
"""
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

# The exact config both shipped bundles use (see module docstring) -- this
# reimplementation is only valid for these values. compress_mode isn't kept
# as an attribute on MaskHeadV6 (only used to build self.compress in
# __init__), so it's checked structurally instead -- see __init__ below.
_EXPECTED_KWARGS = dict(
    tau_arch='linear', tau_feats='no_hubert', offset_prior=False,
    no_compress=False, center_context=False,
    use_ste=False, feature_pool=False, detach_routing=False,
    tau_detach=False, tau_cap=None,
)


def _stack_conv(layers, dim0_key='weight'):
    """Concatenate K conv layers' weight (or bias) tensors along dim 0 --
    exactly the layout a grouped conv with groups multiplied by K expects,
    since each member's own (out, in/groups, k...) block stays contiguous.
    """
    return torch.cat([getattr(l, dim0_key) for l in layers], dim=0)


def _stack_linear_weight(layers):
    """(K, out, in) -- ready for a batched matmul (einsum), one distinct
    weight matrix per member."""
    return torch.stack([l.weight for l in layers], dim=0)


def _stack_linear_bias(layers):
    """(K, 1, out) -- broadcasts over the beat-batch dim in the einsum result."""
    return torch.stack([l.bias for l in layers], dim=0).unsqueeze(1)


def _stack_bn(layers):
    """(weight, bias, running_mean, running_var), each (K, 1, C) -- ready to
    broadcast over the beat-batch dim for the eval-mode affine transform.
    """
    w = torch.stack([l.weight for l in layers], dim=0).unsqueeze(1)
    b = torch.stack([l.bias for l in layers], dim=0).unsqueeze(1)
    m = torch.stack([l.running_mean for l in layers], dim=0).unsqueeze(1)
    v = torch.stack([l.running_var for l in layers], dim=0).unsqueeze(1)
    return w, b, m, v


def _bn_apply(x, stats, eps):
    """x: (K, N, C) or (K, N, C, ...); stats: (weight, bias, mean, var), each
    (K, 1, C[, 1...]) already shaped to broadcast against x."""
    w, b, m, v = stats
    return (x - m) / torch.sqrt(v + eps) * w + b


class GroupedQRSEnsemble:
    """Hand-stacked, grouped-convolution/batched-matmul reimplementation of
    QRSEnsemble.predict() -- see module docstring. Same output contract."""

    _EPS = 1e-5   # nn.BatchNorm{1,2}d's default eps; qrs_model.py never overrides it

    def __init__(self, heads, provenance, manifest, device):
        h0 = heads[0]
        for k, expected in _EXPECTED_KWARGS.items():
            got = getattr(h0, k)
            if got != expected:
                raise ValueError(
                    f"GroupedQRSEnsemble is hardcoded for {k}={expected!r}, "
                    f"but this bundle's heads have {k}={got!r}. This bundle "
                    "needs different stacking logic, not just different weights "
                    "-- see this module's docstring."
                )
        if len(h0.compress) != 10:
            raise ValueError(
                "GroupedQRSEnsemble is hardcoded for compress_mode='full' "
                f"(10-layer compress Sequential), but this bundle's heads have "
                f"{len(h0.compress)} layers -- see this module's docstring."
            )

        self.provenance = list(provenance)
        self.manifest = dict(manifest)
        self.device = torch.device(device)
        self.K = len(heads)
        self.window_size = h0.window_size
        self.C = int(h0.FILM_CHANNELS * h0.scale)
        self.SC = max(1, round(h0.scorer_scale * self.C))
        self.register_t_grid = h0.t_grid.to(self.device)   # (1, W), identical across members by construction

        self._stack_weights(heads)

    # ------------------------------------------------------------- stacking
    def _stack_weights(self, heads):
        K = self.K

        # onset/offset backbone: 6x Conv1d(groups=2) + GELU, alternating.
        # _make_backbone() in qrs_model.py has GELU at odd indices (1,3,5,7,9,11).
        def stack_seq_convs(attr, indices):
            """indices: positions of the Conv1d/Conv2d layers within the Sequential."""
            out = []
            for i in indices:
                layers = [getattr(h, attr)[i] for h in heads]
                out.append((_stack_conv(layers, 'weight'), _stack_conv(layers, 'bias')))
            return out

        self._onset_backbone_w = stack_seq_convs('onset_backbone', (0, 2, 4, 6, 8, 10))
        self._offset_backbone_w = stack_seq_convs('offset_backbone', (0, 2, 4, 6, 8, 10))
        self._onset_scorer_w = stack_seq_convs('onset_scorer', (0, 2, 4, 6))
        self._offset_scorer_w = stack_seq_convs('offset_scorer', (0, 2, 4, 6))

        # compress: Conv2d, BatchNorm2d, GELU, Conv2d, GELU, Conv2d, GELU, Conv2d, GELU, Conv2d
        # indices:     0        1         2     3      4     5      6     7      8     9
        compress_conv_idx = (0, 3, 5, 7, 9)
        self._compress_w = []
        for i in compress_conv_idx:
            layers = [h.compress[i] for h in heads]
            self._compress_w.append((_stack_conv(layers, 'weight'), _stack_conv(layers, 'bias')))
        self._compress_bn = _stack_bn([h.compress[1] for h in heads])

        # tau heads: BatchNorm1d(TAU_IN) -> Linear(TAU_IN, 1). tau_arch='linear'
        # means exactly 2 layers, indices 0 (BN) and 1 (Linear).
        self._onset_tau_bn = _stack_bn([h.onset_tau_head[0] for h in heads])
        self._onset_tau_lin_w = _stack_linear_weight([h.onset_tau_head[1] for h in heads])
        self._onset_tau_lin_b = _stack_linear_bias([h.onset_tau_head[1] for h in heads])
        self._offset_tau_bn = _stack_bn([h.offset_tau_head[0] for h in heads])
        self._offset_tau_lin_w = _stack_linear_weight([h.offset_tau_head[1] for h in heads])
        self._offset_tau_lin_b = _stack_linear_bias([h.offset_tau_head[1] for h in heads])

        # LeadFeatureCompressor: BatchNorm1d(L*2C) -> Linear(L*2C,128) -> GELU -> Linear(128,256)
        def stack_lead_compressor(attr):
            comps = [getattr(h, attr) for h in heads]
            bn = _stack_bn([c.norm for c in comps])
            w1 = _stack_linear_weight([c.proj[0] for c in comps])
            b1 = _stack_linear_bias([c.proj[0] for c in comps])
            w2 = _stack_linear_weight([c.proj[2] for c in comps])
            b2 = _stack_linear_bias([c.proj[2] for c in comps])
            return bn, w1, b1, w2, b2

        self._lead_comp_on = stack_lead_compressor('lead_feat_compressor_on')
        self._lead_comp_off = stack_lead_compressor('lead_feat_compressor_off')

        # tau_c: per-member learned scalar prior, (K, 1) to broadcast over N.
        self._tau_c_on = torch.stack([h.tau_c_on for h in heads]).view(K, 1)
        self._tau_c_off = torch.stack([h.tau_c_off for h in heads]).view(K, 1)

    @classmethod
    def from_ensemble(cls, ensemble):
        """Wrap an already-loaded qrs_ensemble.QRSEnsemble."""
        return cls(ensemble.heads, ensemble.provenance, ensemble.manifest, ensemble.device)

    def __len__(self):
        return self.K

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _norm(x):
        mu = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (x - mu) / std

    def _conv1d_reflect(self, x, weight, bias, padding, dilation, groups):
        """qrs_model.py's backbone/scorer Conv1d layers use padding_mode='reflect';
        F.conv1d has no such kwarg, so pad manually before a zero-padded conv."""
        if padding > 0:
            x = F.pad(x, (padding, padding), mode='reflect')
        return F.conv1d(x, weight, bias, padding=0, dilation=dilation, groups=groups)

    # ---------------------------------------------------------------- run
    @torch.no_grad()
    def predict(self, window, emb, routing_alpha=1.0, return_members=False):
        """Same contract as QRSEnsemble.predict() -- see that docstring."""
        window = torch.as_tensor(window).to(self.device)
        emb = torch.as_tensor(emb).to(self.device)
        K = self.K
        N, L, _, W = window.shape
        C = self.C

        # ── backbone (grouped conv1d, reflect padding, dilations 1..32) ──
        w = window.reshape(N * L, 2, W)
        w = torch.stack([w[:, 0], w[:, 1].abs()], dim=1)
        f = self._norm(w)
        f_rep = f.repeat(1, K, 1)   # (N*L, K*2, W)

        def run_backbone(stacked):
            x = f_rep
            dilations = (1, 2, 4, 8, 16, 32)
            for i, (wt, bs) in enumerate(stacked):
                d = dilations[i]
                x = self._conv1d_reflect(x, wt, bs, padding=d, dilation=d, groups=K * 2)
                x = F.gelu(x)
            return x

        h_on_flat = self._norm(run_backbone(self._onset_backbone_w))    # (N*L, K*C, W)
        h_off_flat = self._norm(run_backbone(self._offset_backbone_w))

        # ── scorer (grouped conv1d, plain zero padding, groups=K) ────────
        def run_scorer(x, stacked):
            paddings = (3, 1, 1, 1)
            for i, (wt, bs) in enumerate(stacked):
                x = self._conv1d_reflect(x, wt, bs, padding=paddings[i], dilation=1, groups=K)
                if i < len(stacked) - 1:
                    x = F.gelu(x)
            return x   # (N*L, K*1, W) == (N*L, K, W)

        score_on = run_scorer(h_on_flat, self._onset_scorer_w)     # (N*L, K, W)
        score_off = run_scorer(h_off_flat, self._offset_scorer_w)  # offset_prior=False -- no bias added

        t_grid = self.register_t_grid   # (1, W)
        attn_on_flat = F.softmax(score_on, dim=-1)
        t_on_flat = (attn_on_flat * t_grid.unsqueeze(0)).sum(-1)     # (N*L, K)
        attn_off_flat = F.softmax(score_off, dim=-1)
        t_off_flat = (attn_off_flat * t_grid.unsqueeze(0)).sum(-1)   # (N*L, K)

        # per-lead, K as the leading axis from here on
        attn_on_per_lead = attn_on_flat.view(N, L, K, W).permute(2, 0, 1, 3)    # (K,N,L,W)
        attn_off_per_lead = attn_off_flat.view(N, L, K, W).permute(2, 0, 1, 3)
        t_on_leads = t_on_flat.view(N, L, K).permute(2, 0, 1)                    # (K,N,L)
        t_off_leads = t_off_flat.view(N, L, K).permute(2, 0, 1)

        # ── routing / compress ────────────────────────────────────────────
        # Layer 0 alone consumes the RAW shared 768-channel embedding, not
        # yet split into per-member blocks -- since it's ungrouped (groups=1)
        # per member to begin with, a regular (groups=1) conv over the K*C
        # stacked output channels already computes "K members' independent
        # transforms of the same shared input" directly: each output row
        # already carries its own full 768-wide weight (see _stack_conv),
        # so groups=1 + the ORIGINAL un-repeated input gives the identical
        # result to groups=K + `x.repeat(1, K, 1, 1)` without ever
        # materializing that K*768-channel repeated tensor (~2.9GB at
        # B=64, K=32 -- the actual OOM cause, not the backbone's small
        # Cin=2 repeat). Layers 1-4 consume the now K*C-wide, per-member
        # BLOCK-structured output of layer 0 and must stay grouped at
        # groups=K (each member's own C-channel block may only feed that
        # member's own weights, matching the original per-member model);
        # they're unaffected by this change.
        ctx = emb   # center_context=False
        x = ctx.permute(0, 3, 1, 2)          # (N, D, L, T) -- NOT repeated
        wt, bs = self._compress_w[0]
        x = F.conv2d(x, wt, bs, groups=1)                    # (N, K*C, L, T)
        # BatchNorm2d applied directly on (N, K*C, L, T) -- stats broadcast
        # over N, L, T via shape (1, K*C, 1, 1).
        x = self._bn2d(x, self._compress_bn)
        x = F.gelu(x)
        for i in (1, 2, 3):
            wt, bs = self._compress_w[i]
            x = F.conv2d(x, wt, bs, stride=(1, 2), padding=(0, 1), groups=K)
            x = F.gelu(x)
        wt, bs = self._compress_w[4]
        x = F.conv2d(x, wt, bs, groups=K)    # (N, K*2, L, 1)
        g = x.squeeze(-1).view(N, K, 2, L)   # (N,K,2,L)

        w_on = F.softmax(g[:, :, 0, :] * routing_alpha, dim=-1).permute(1, 0, 2)   # (K,N,L)
        w_off = F.softmax(g[:, :, 1, :] * routing_alpha, dim=-1).permute(1, 0, 2)

        # ── tau (BatchNorm1d + Linear via batched matmul, groups implicit) ─
        h_on_leads = h_on_flat.view(N, L, K, C, W).permute(2, 0, 1, 3, 4)    # (K,N,L,C,W)
        h_off_leads = h_off_flat.view(N, L, K, C, W).permute(2, 0, 1, 3, 4)

        def tau_from_all(attn_per_lead, t_leads, h_leads, w_route, lead_comp, tau_bn, tau_lin_w, tau_lin_b, tau_c):
            Wsz = self.window_size
            pooled_uni = attn_per_lead.mean(dim=2)   # (K,N,W)  -- mean over L (dim=2 here)
            H_pooled_norm = (-(pooled_uni * pooled_uni.clamp(min=1e-10).log()).sum(-1)
                             / math.log(Wsz)).unsqueeze(-1)                          # (K,N,1)
            H_l = -(attn_per_lead * attn_per_lead.clamp(min=1e-10).log()).sum(-1)     # (K,N,L)
            H_mean = (H_l.mean(dim=-1) / math.log(Wsz)).unsqueeze(-1)                # (K,N,1)
            H_std = (H_l.std(dim=-1) / math.log(Wsz)).unsqueeze(-1)                  # (K,N,1)
            var_t = (t_leads.var(dim=-1).clamp(min=0) / (Wsz ** 2)).unsqueeze(-1)    # (K,N,1)
            peak_leads = attn_per_lead.max(dim=-1).values.mean(dim=-1, keepdim=True) # (K,N,1)
            hand_feat = torch.cat([H_pooled_norm, H_mean, H_std, var_t, peak_leads], dim=-1)  # (K,N,5)

            # LeadFeatureCompressor: mean+max pool over W, concat, BN1d, Linear->GELU->Linear
            bn, w1, b1, w2, b2 = lead_comp
            mean_p = h_leads.mean(dim=-1)     # (K,N,L,C)
            max_p = h_leads.amax(dim=-1)      # (K,N,L,C)
            pooled = torch.cat([mean_p, max_p], dim=-1).flatten(2)   # (K,N,L*2C)
            pooled = _bn_apply(pooled, bn, self._EPS)
            h1 = F.gelu(torch.einsum('koi,kni->kno', w1, pooled) + b1)   # (K,N,128)
            feat_lead = torch.einsum('koi,kni->kno', w2, h1) + b2        # (K,N,256)

            feat = torch.cat([feat_lead, hand_feat], dim=-1)   # (K,N,261) -- tau_feats='no_hubert'
            feat = _bn_apply(feat, tau_bn, self._EPS)
            tau_b = torch.einsum('koi,kni->kno', tau_lin_w, feat).squeeze(-1) + tau_lin_b.squeeze(-1)  # (K,N)
            tau = (tau_c + tau_b).clamp(min=1.0)   # tau_cap=None -- no cap
            return tau

        tau_on = tau_from_all(attn_on_per_lead, t_on_leads, h_on_leads, w_on,
                               self._lead_comp_on, self._onset_tau_bn,
                               self._onset_tau_lin_w, self._onset_tau_lin_b, self._tau_c_on)
        tau_off = tau_from_all(attn_off_per_lead, t_off_leads, h_off_leads, w_off,
                                self._lead_comp_off, self._offset_tau_bn,
                                self._offset_tau_lin_w, self._offset_tau_lin_b, self._tau_c_off)

        # ── pooled attention -> t_on/t_off (feature_pool=False branch) ───
        pooled_attn_on = (w_on.unsqueeze(-1) * attn_on_per_lead).sum(dim=2)    # (K,N,W)
        pooled_attn_off = (w_off.unsqueeze(-1) * attn_off_per_lead).sum(dim=2)
        t_on = (pooled_attn_on * t_grid.unsqueeze(0)).sum(-1)    # (K,N)
        t_off = (pooled_attn_off * t_grid.unsqueeze(0)).sum(-1)

        C_out = torch.stack([t_on, tau_on, t_off, tau_off], dim=-1)   # (K,N,4)
        C_np = C_out.float().cpu().numpy()

        on = np.median(C_np[:, :, 0], axis=0)
        off = np.median(C_np[:, :, 2], axis=0)
        res = {
            "onset": on,
            "offset": off,
            "duration": off - on,
            "tau_on": np.median(C_np[:, :, 1], axis=0),
            "tau_off": np.median(C_np[:, :, 3], axis=0),
            "spread_on": C_np[:, :, 0].std(axis=0, ddof=1) if K > 1 else np.zeros_like(on),
            "spread_off": C_np[:, :, 2].std(axis=0, ddof=1) if K > 1 else np.zeros_like(off),
            "n_members": K,
        }
        if return_members:
            res["members"] = C_np
        return res

    @staticmethod
    def _bn2d(x, stats):
        w, b, m, v = stats
        shape = (1, -1, 1, 1)
        return ((x - m.view(*shape)) / torch.sqrt(v.view(*shape) + GroupedQRSEnsemble._EPS)
                * w.view(*shape) + b.view(*shape))


def _self_test():
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
        grouped = GroupedQRSEnsemble.from_ensemble(ens)

        slow_out = ens.predict(g['window'], g['emb'])
        fast_out = grouped.predict(g['window'], g['emb'])

        for k in ('onset', 'offset', 'duration', 'tau_on', 'tau_off'):
            d_vs_slow = np.abs(fast_out[k] - slow_out[k]).max()
            d_vs_golden = np.abs(fast_out[k] - g[k]).max()
            print(f"  {k:10s} grouped-vs-loop={d_vs_slow:.6f}  grouped-vs-golden={d_vs_golden:.4f}")
            assert d_vs_slow < 0.5, f"{bundle}/{k}: grouped diverges from the original loop by {d_vs_slow}"
        print("  OK")

    if not ran_any:
        print("no bundle+golden pairs found next to this file -- nothing to test")


if __name__ == '__main__':
    _self_test()
