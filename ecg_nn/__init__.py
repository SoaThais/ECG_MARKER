"""ecg_nn — ECG beat detection + QRS inference.

Public API
----------
from ecg_nn import Recording

rec = Recording.from_signal(leads)                 # leads: dict[str, np.ndarray]
rec = Recording.from_signal(leads, predict=True)   # runs NN inference too

rec.beats          # list[Beat] — all detected beats
rec.annotated      # beats with qrs_duration + qt_interval filled
rec.unannotated    # beats without QRS/QT predictions
rec.noisy          # beats whose window overlaps a neighbour
rec[i]             # Beat at index i
len(rec)           # total beat count

# ECG_MARKER-compatible export
rec.to_ecg_marker_qrs()   # list of [start_ms, end_ms, period_ms, duration_ms]
rec.to_ecg_marker_qt()    # list of [start_ms, end_ms, period_ms, duration_ms]

Direct access to the model, for scripts that already have windows + embeddings
and do not need beat extraction:

    from ecg_nn import load_optimized, BUNDLE_DIR
    ens = load_optimized(os.path.join(BUNDLE_DIR, "megamodel_loo32.pt"))
    out = ens.predict(window, emb)

Layout
------
Flat by design -- see README.md. `qrs_model.py` and `qrs_ensemble.py` are
vendored from the training repo and must stay byte-identical so a re-sync is a
straight copy; they import each other by bare name, which is why this file puts
its own directory on sys.path before importing them.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

#: Where the .pt bundles and golden .npz fixtures live.
BUNDLE_DIR = os.path.join(_HERE, 'weights')

from .recording import Recording          # noqa: E402

__all__ = ['Recording', 'BUNDLE_DIR', 'load_optimized', 'run_golden',
           'run_golden_optimized', 'QRSEnsemble', 'OptimizedQRSEnsemble']


def __getattr__(name):
    """Expose the inference API lazily.

    Everything below pulls in torch, so importing it eagerly would make
    `import ecg_nn` (and therefore the ECG_MARKER GUI's startup) pay for a
    torch import even when the user never runs inference.
    """
    if name in ('QRSEnsemble', 'run_golden'):
        import qrs_ensemble
        return getattr(qrs_ensemble, name)
    if name in ('OptimizedQRSEnsemble', 'load_optimized', 'run_golden_optimized'):
        import qrs_ensemble_optimized
        return getattr(qrs_ensemble_optimized, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
