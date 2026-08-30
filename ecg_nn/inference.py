"""Inference utilities — convert model mask output to interval estimates."""

import numpy as np


def mask_to_interval(mask_1d, lo_thr=0.1, hi_thr=0.9):
    """Convert a soft probability mask to start/end point estimates with uncertainty.

    The onset  transition (0→1) width gives start uncertainty.
    The offset transition (1→0) width gives end   uncertainty.

    Parameters
    ----------
    mask_1d     : array-like, shape (W,)   per-ms sigmoid probability
    lo_thr      : float   outer threshold (default 0.1)
    hi_thr      : float   inner threshold (default 0.9)

    Returns
    -------
    start        : float | None   onset  midpoint in samples (window-relative)
    start_uncert : float          half-width of onset  transition in samples
    end          : float | None   offset midpoint in samples (window-relative)
    end_uncert   : float          half-width of offset transition in samples

    Notes
    -----
    All values are in samples (= ms at 1 kHz).  Add window_start_ms to convert
    to absolute signal positions.

    Returns (None, 0, None, 0) if the mask never crosses 0.5.
    """
    if hasattr(mask_1d, 'cpu'):
        mask_1d = mask_1d.cpu().numpy()
    mask_1d = np.asarray(mask_1d, dtype=float).ravel()

    above_lo = np.where(mask_1d > lo_thr)[0]
    above_hi = np.where(mask_1d > hi_thr)[0]

    if len(above_lo) == 0:
        return None, 0.0, None, 0.0

    # ── onset: first crossing of lo_thr then hi_thr ──────────────────────────
    onset_lo = float(above_lo[0])
    onset_hi = float(above_hi[0]) if len(above_hi) > 0 else onset_lo

    start        = (onset_lo + onset_hi) / 2.0
    start_uncert = (onset_hi - onset_lo) / 2.0

    # ── offset: last crossing of hi_thr then lo_thr ──────────────────────────
    offset_hi = float(above_hi[-1]) if len(above_hi) > 0 else float(above_lo[-1])
    offset_lo = float(above_lo[-1])

    end        = (offset_hi + offset_lo) / 2.0
    end_uncert = (offset_lo - offset_hi) / 2.0

    return start, start_uncert, end, end_uncert


def mask_confidence(mask_1d):
    """Scalar confidence score for a mask: mean distance from 0.5, normalised to [0, 1].

    1.0 = perfectly binary (all samples at 0 or 1).
    0.0 = maximally uncertain (all samples at 0.5).
    """
    if hasattr(mask_1d, 'cpu'):
        mask_1d = mask_1d.cpu().numpy()
    mask_1d = np.asarray(mask_1d, dtype=float).ravel()
    return float(np.mean(np.abs(mask_1d - 0.5)) * 2.0)
