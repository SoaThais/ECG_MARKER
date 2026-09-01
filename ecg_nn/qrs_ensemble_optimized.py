"""Fastest verified inference path for the QRS delineation ensemble.

This is what the app should use. It is `DeferredSyncQRSEnsemble` (one GPU->CPU
sync instead of K) plus two things measured to stack on top of it:

  1. torch.compile(mode='max-autotune')  -- fuses the reflect-pad / bias /
     GELU chain between convolutions into single Triton kernels.
  2. fp16 for every submodule that tolerates it (see FP16_SAFE below); the
     two backbones stay fp32 because they do not.

Measured, RTX 4070 Ti SUPER, torch 2.6.0+cu124, megamodel_loo32 (K=32), B=32:

    deferred fp32 (the previous default)  122.0 ms   3.81 ms/beat   1.00x
    + compile                              66.4 ms   2.08 ms/beat   1.84x
    + fp16-safe modules (no compile)      108.6 ms   3.40 ms/beat   1.12x
    + both  (this module)                  58.4 ms   1.83 ms/beat   2.09x

Golden fixture (12 real beats, megamodel_loo32), worst-case |diff| against the
recorded fp32 outputs: onset 0.079 ms, offset 0.130, duration 0.107, tau_on
0.086, tau_off 0.057 -- against run_golden()'s atol of 0.5 ms, so ~4x margin
on the worst metric. Verify with `python qrs_ensemble_optimized.py` (self-test
below) whenever weights, torch, or the driver change.

Why fp16 stops at the backbone
------------------------------
Casting either backbone to fp16 moves onset by ~23 ms -- 46x outside the
golden tolerance -- while every other module is fine. Casting only the
backbone's LAST layer is just as bad, so it is not error accumulated over its
six dilated convs. Two natural explanations were tested and are both wrong:

  - Not overflow. Activations shrink monotonically through the backbone
    (max|x| 120 -> 61 -> 14 -> 4.8 -> 1.2 -> 0.43 -> 0.16), nowhere near
    fp16's 65504.
  - Not subnormal underflow. Median |x| is ~0.05 and at most 1.4% of values
    fall below fp16's smallest normal (6.1e-5) at any layer.
  - bf16 is not the answer either: it is worse everywhere, and breaks even
    the configuration fp16 passes (0.78 ms onset). So the binding constraint
    is mantissa width, where fp16's 11 bits beat bf16's 8.

What remains is that a ~1e-3 relative perturbation in the backbone is
amplified far more by the soft-argmax than the same perturbation in the
scorer, which sits closer to it. That asymmetry is not explained. The
boundary below is therefore empirical: it is where the golden fixture says
it is, not where a theory of the numerics says it should be. Re-run the
self-test rather than reasoning about it if you change the architecture.

Approaches that did NOT work (see dev/attic/, kept as reference impls)
----------------------------------------------------------------------
vmap (0.96x), CUDA graphs (1.03x), and hand-written grouped convolution
(measured a net LOSS at production shapes) all attacked kernel launch count.
That was the wrong target: profiling showed pointwise ops (reflect-pad, GELU,
bias) were 59.6% of GPU time and the convolutions only 33%, so fusion -- not
scheduling -- was the lever. The grouped path additionally fails because
cuDNN decomposes a grouped convolution back into one launch per group
(129 launches for a groups=64 layer) whenever channels-per-group > 1: a
grouped conv is a block-diagonal GEMM, and the fp32 implicit-GEMM kernels
cannot tile that without doing K-times the arithmetic on zeros.
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from qrs_ensemble import QRSEnsemble                      # noqa: E402
from qrs_ensemble_deferred import DeferredSyncQRSEnsemble  # noqa: E402

# Submodules of MaskHeadV6 that pass the golden fixture in fp16. The two
# backbones (onset_backbone / offset_backbone) are deliberately absent -- see
# the module docstring. Names are attributes on MaskHeadV6; a bundle whose
# architecture lacks one is skipped rather than erroring, so this stays valid
# for the no-HuBERT tau head both shipped bundles use (where
# routing_compressor_* exist but are never called).
FP16_SAFE = (
    "onset_scorer", "offset_scorer",
    "compress",
    "lead_feat_compressor_on", "lead_feat_compressor_off",
    "onset_tau_head", "offset_tau_head",
)

FP32_ONLY = ("onset_backbone", "offset_backbone")


def _cast_module_fp16(mod):
    """Run `mod` in fp16 while keeping its external contract in fp32.

    Weights/buffers are cast in place; a pre-hook casts incoming floating
    tensors down and a forward hook casts the output back up, so the
    surrounding fp32 graph (and the soft-argmax in particular) is untouched.
    Cheaper than an autocast region and far more selective: autocast would
    also take the backbones, which is exactly what must not happen.
    """
    mod.half()

    def _down(m, args):
        return tuple(a.half() if torch.is_tensor(a) and a.is_floating_point() else a
                     for a in args)

    def _up(m, args, out):
        return out.float() if torch.is_tensor(out) else out

    mod.register_forward_pre_hook(_down)
    mod.register_forward_hook(_up)


class OptimizedQRSEnsemble:
    """Drop-in for QRSEnsemble.predict() -- identical dict contract.

    Construct via from_ensemble(). NOTE this mutates the QRSEnsemble it is
    handed (the fp16 casts are applied in place to its heads), so do not keep
    using the original afterwards; load a fresh one if you need fp32.
    """

    #: Minimum CUDA compute capability Triton (inductor's GPU backend) supports.
    #: Below this, torch.compile is skipped outright -- a GTX 1080 is 6.1.
    _MIN_TRITON_CC = (7, 0)

    def __init__(self, deferred, compiled_predict, fp16, compiled):
        self._deferred = deferred
        self._predict = compiled_predict
        self.fp16 = fp16
        self.compiled = compiled
        # torch.compile is LAZY: the backend does not run until the first
        # forward call, so a backend that cannot support this GPU raises
        # there, not at construction. Guard the first call and fall back
        # permanently if it does. After one successful call, stop catching --
        # a later exception is a real error and must surface.
        self._compile_unverified = compiled
        self.device = deferred.device
        self.provenance = deferred.provenance
        self.manifest = deferred.manifest

    @classmethod
    def from_ensemble(cls, ensemble, fp16=True, compile=True):
        """Wrap a loaded QRSEnsemble.

        fp16/compile are requests, not guarantees: both are silently skipped
        on CPU (there is no benefit and compile's autotuning is CUDA-shaped),
        and compile is skipped if torch.compile is unavailable. Check the
        .fp16 / .compiled attributes for what actually happened.
        """
        on_cuda = ensemble.device.type == "cuda"
        fp16 = bool(fp16) and on_cuda
        if fp16:
            for h in ensemble.heads:
                for name in FP16_SAFE:
                    mod = getattr(h, name, None)
                    if mod is not None:
                        _cast_module_fp16(mod)

        deferred = DeferredSyncQRSEnsemble.from_ensemble(ensemble)
        predict = deferred.predict
        compiled = False
        if compile and on_cuda and hasattr(torch, "compile") and cls._triton_capable():
            try:
                predict = torch.compile(deferred.predict, mode="max-autotune")
                compiled = True
            except Exception:
                # A compile failure must never cost correctness -- fall back
                # to the eager deferred path, which is still the 1.12x fp16
                # win on its own. (This catches construction-time failures;
                # the far more likely first-call failure is handled in
                # predict() -- torch.compile is lazy.)
                predict = deferred.predict
        return cls(deferred, predict, fp16, compiled)

    @classmethod
    def _triton_capable(cls):
        """True when this GPU is new enough for inductor's Triton backend.

        Checked up front because the alternative is discovering it as an
        exception midway through the first batch of a real recording. Pascal
        (GTX 1080, CC 6.1) is below the cutoff; Turing and later are above.
        """
        try:
            return torch.cuda.get_device_capability() >= cls._MIN_TRITON_CC
        except Exception:
            return False

    def __len__(self):
        return len(self._deferred)

    def predict(self, window, emb, routing_alpha=1.0, return_members=False):
        if self._compile_unverified:
            try:
                out = self._predict(window, emb, routing_alpha=routing_alpha,
                                    return_members=return_members)
            except Exception:
                # First call, so this is the compile backend failing (it runs
                # lazily). Drop to eager for the rest of this object's life
                # and retry -- correctness must not depend on inductor.
                self._predict = self._deferred.predict
                self.compiled = False
                self._compile_unverified = False
                return self._deferred.predict(window, emb, routing_alpha=routing_alpha,
                                              return_members=return_members)
            self._compile_unverified = False
            return out
        return self._predict(window, emb, routing_alpha=routing_alpha,
                             return_members=return_members)

    def check_encoder(self, encoder_state_dict):
        return self._deferred.check_encoder(encoder_state_dict)


def load_optimized(bundle_path, device="cuda", fp16=True, compile=True):
    """QRSEnsemble.load() + from_ensemble(), the one call the app needs."""
    return OptimizedQRSEnsemble.from_ensemble(
        QRSEnsemble.load(bundle_path, device=device), fp16=fp16, compile=compile)


def run_golden_optimized(bundle_path, golden_path, atol=0.5, **kw):
    """run_golden() for this path: same fixtures, same tolerance.

    Kept separate from qrs_ensemble.run_golden() because that one constructs
    its own plain QRSEnsemble; this replays the fixtures through the fp16 +
    compiled path that actually ships, which is the thing that needs pinning.
    """
    g = np.load(golden_path)
    ens = load_optimized(bundle_path, **kw)
    out = ens.predict(g["window"], g["emb"])
    bad = []
    for k in ("onset", "offset", "duration", "tau_on", "tau_off"):
        d = np.abs(out[k] - g[k]).max()
        if d > atol:
            bad.append("%s max|diff|=%.3g" % (k, d))
    if bad:
        raise AssertionError("golden mismatch: " + "; ".join(bad))
    return True


def main():
    """Self-test: replay the shipped golden fixtures through this path.

    Requires the bundles in weights/. Skips cleanly when they are absent
    or when there is no GPU, since on CPU this module is by construction
    just DeferredSyncQRSEnsemble.
    """
    bundle_dir = os.path.join(_HERE, "weights")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    found = False
    for name in ("megamodel_loo32", "allseed64"):
        b = os.path.join(bundle_dir, name + ".pt")
        g = os.path.join(bundle_dir, "golden_" + name + ".npz")
        if not (os.path.exists(b) and os.path.exists(g)):
            continue
        found = True
        ens = load_optimized(b, device=device)
        gold = np.load(g)
        out = ens.predict(gold["window"], gold["emb"])
        worst = {k: float(np.abs(out[k] - gold[k]).max())
                 for k in ("onset", "offset", "duration", "tau_on", "tau_off")}
        print("%-18s K=%-3d fp16=%-5s compiled=%-5s device=%s"
              % (name, len(ens), ens.fp16, ens.compiled, device))
        print("   " + "  ".join("%s %.4g" % kv for kv in worst.items()))
        assert max(worst.values()) <= 0.5, worst
        print("   golden OK (atol 0.5 ms)")
    if not found:
        print("no bundles in %s -- nothing to check" % bundle_dir)
        return
    print("qrs_ensemble_optimized self-test OK")


if __name__ == "__main__":
    main()
