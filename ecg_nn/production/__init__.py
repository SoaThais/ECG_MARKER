"""Production QRS delineation ensemble: vendored model, converter, inference.

Contents
--------
    qrs_model.py               vendored MaskHeadV6 -- do not edit
    qrs_ensemble.py            vendored loader + reference predict() -- do not edit
    qrs_ensemble_deferred.py   one sync instead of K (1.08x), used as the base
    qrs_ensemble_optimized.py  what the app runs: + compile + fp16 (2.09x)
    package_ensemble.py        converter: run directories -> a .pt bundle
    make_golden.py             records the golden fixtures for a bundle
    regen_golden.py            re-records them after an intentional change

The two vendored files ship exactly as pulled from daint and are marked "do
not edit" in README.md; everything added on top lives beside them rather than
patching them. They use bare imports of each other (`from qrs_model import
MaskHeadV6`), which is why this __init__ puts its own directory on sys.path
before importing -- that keeps them byte-identical to the daint dump, so a
re-sync is a straight copy.

Weights are NOT here. The .pt bundles and .npz golden fixtures are 145MB and
stay in models/production/ (BUNDLE_DIR below) so the package stays light.

    from ecg_nn.production import load_optimized, BUNDLE_DIR
    ens = load_optimized(os.path.join(BUNDLE_DIR, "megamodel_loo32.pt"))
    out = ens.predict(window, emb)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from qrs_ensemble import QRSEnsemble, run_golden                    # noqa: E402
from qrs_ensemble_deferred import DeferredSyncQRSEnsemble           # noqa: E402
from qrs_ensemble_optimized import (                                # noqa: E402
    OptimizedQRSEnsemble, load_optimized, run_golden_optimized,
    FP16_SAFE, FP32_ONLY,
)

#: Where the .pt / .npz files live (deliberately outside the package).
BUNDLE_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "models", "production"))

__all__ = [
    "QRSEnsemble", "DeferredSyncQRSEnsemble", "OptimizedQRSEnsemble",
    "load_optimized", "run_golden", "run_golden_optimized",
    "FP16_SAFE", "FP32_ONLY", "BUNDLE_DIR",
]
