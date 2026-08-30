"""ecg_nn — ECG beat detection + QRS/QT inference package.

Public API
----------
from ecg_nn import Recording

rec = Recording.from_signal(leads)           # leads: dict[str, np.ndarray]
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
"""

from .recording import Recording

__all__ = ['Recording']
