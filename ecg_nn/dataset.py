"""PyTorch Dataset for Beat objects with optional HuBERT-ECG preprocessing."""

import copy

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import resample

from .beat import (FS, Beat, WINDOW_PRE, WINDOW_POST, WINDOW_SIZE,
                  CONTEXT_PRE, CONTEXT_POST, CONTEXT_SHIFT_MAX)

# =========================================================
# HuBERT-ECG target spec
# =========================================================
HUBERT_FS        = 500    # Hz — model input sample rate
HUBERT_N_SAMPLES = 2500   # samples (5 seconds)


def preprocess_hubert(context_window):
    """
    Transform a 5-second context window into HuBERT format (MULTI-LEAD).

    Input:
        (13, CONTEXT_PRE+CONTEXT_POST) at FS (1 kHz) — centred on the spike

    Output:
        (12, 2500) at HUBERT_FS (500 Hz)
    """
    leads = []
    for lead in context_window:
        resampled = resample(lead, HUBERT_N_SAMPLES).astype(np.float32)
        mu = resampled.mean()
        std = resampled.std()
        leads.append((resampled - mu) / (std + 1e-6))

    return np.stack(leads, axis=0)[:-1, :]  # drop VD d → (12, 2500)


# =========================================================
# Mask builder
# =========================================================

def build_mask(beat):
    """Build a (2, WINDOW_SIZE) float32 binary mask from beat annotations.

    Channel 0: QRS interval  [qrs_start, qrs_start + qrs_duration)
    Channel 1: QT  interval  [qt_start,  qt_start  + qt_interval)

    All positions are in ms (= samples at 1 kHz), clipped to [0, WINDOW_SIZE).
    Summing along the last axis gives duration in ms.
    """
    mask         = np.zeros((2, WINDOW_SIZE), dtype=np.float32)
    window_start = beat.spike_idx - beat.window_pre   # absolute ms of window[0]

    if beat.qrs_start is not None and beat.qrs_duration is not None:
        lo = int(round(beat.qrs_start))    - window_start
        hi = lo + int(round(beat.qrs_duration))
        lo, hi = max(0, lo), min(WINDOW_SIZE, hi)
        if lo < hi:
            mask[0, lo:hi] = 1.0

    if beat.qt_start is not None and beat.qt_interval is not None:
        lo = int(round(beat.qt_start))     - window_start
        hi = lo + int(round(beat.qt_interval))
        lo, hi = max(0, lo), min(WINDOW_SIZE, hi)
        if lo < hi:
            mask[1, lo:hi] = 1.0

    return mask


# =========================================================
# Dataset
# =========================================================

class BeatDataset(Dataset):
    """PyTorch Dataset wrapping a list of Beat objects.

    Parameters
    ----------
    beats        : list[Beat]
    transform    : callable | None
        Applied to each context window before converting to tensor.
        Pass `preprocess_hubert` to use HuBERT-ECG formatting.
    require_both : bool
        If True (default), only beats with both qrs_duration and qt_interval
        are included.
    """

    def __init__(self, beats, transform=None, require_both=True):
        if require_both:
            beats = [b for b in beats
                     if b.qrs_duration is not None and b.qt_interval is not None]
            # Drop beats where the QRS mask sum differs from the annotation by
            # more than 10% — catches T-wave misdetections and heavily clipped beats.
            def _mask_ok(b):
                m = build_mask(b)
                qrs_sum = m[0].sum()
                return qrs_sum >= 0.9 * b.qrs_duration
            n_before = len(beats)
            beats = [b for b in beats if _mask_ok(b)]
            dropped = n_before - len(beats)
            if dropped:
                print(f'  [dataset] dropped {dropped} beat(s) with clipped/mismatched QRS mask')
        self.beats     = beats
        self.transform = transform

    def __len__(self):
        return len(self.beats)

    def __getitem__(self, idx):
        beat   = self.beats[idx]
        window = beat.context_window.astype(np.float32)

        if self.transform is not None:
            window = self.transform(window)

        x = torch.from_numpy(window)
        d = torch.from_numpy(beat.decision_window)          # (W,)
        y = torch.from_numpy(build_mask(beat))              # (2, W)

        return x, d, y


# =========================================================
# Augmentation
# =========================================================

def generate_expansion_scale(beats, n, seed):
    """Generate synthetic beats by per-lead amplitude scaling.

    Each synthetic beat is a shallow copy of a source beat with
    context_window and context_buffer scaled by a per-lead factor
    drawn uniformly from [0.6, 2.0].  Annotations and decision_window
    are unchanged (PT signal is lead-II derived and unaffected by scaling).

    Parameters
    ----------
    beats : list[Beat]  — source pool (must have context_buffer)
    n     : int         — number of synthetic beats to generate
    seed  : int         — RNG seed for reproducibility / caching

    Returns
    -------
    list[Beat]
    """
    rng         = np.random.default_rng(seed)
    src_indices = [i for i, b in enumerate(beats)
                   if getattr(b, 'context_buffer', None) is not None]
    src_beats   = [beats[i] for i in src_indices]

    synthetic = []
    for _ in range(n):
        idb    = rng.integers(len(src_beats))
        src    = src_beats[idb]
        scales = rng.uniform(0.6, 2.0, size=(src.context_buffer.shape[0], 1))

        beat = copy.copy(src)
        beat.parent          = src_indices[idb]  # index into input beats list
        beat.context_buffer  = src.context_buffer * scales
        beat.context_window  = src.context_window * scales
        beat.window          = src.window * scales  # keep display window aligned
        beat.lead_scales     = scales.squeeze()   # (n_leads,) for inspection
        synthetic.append(beat)

    return synthetic


def generate_expansion_shift(beats, n, max_shift, seed):
    """Generate synthetic beats by shifting the context/decision window.

    Each synthetic beat is a shallow copy of a source beat with the
    context and PT windows sliced at a random offset from the buffers.
    window_pre is updated so build_mask produces a correctly shifted label.

    Parameters
    ----------
    beats     : list[Beat]  — source pool (must have context_buffer / pt_buffer)
    n         : int         — number of synthetic beats to generate
    max_shift : int         — max shift magnitude in samples at FS (ms)
    seed      : int         — RNG seed for reproducibility / caching

    Returns
    -------
    list[Beat]
    """
    MAX_REALISTIC_SHIFT = 100  # beyond detector jitter (±20ms refinement + margin)
    assert max_shift <= MAX_REALISTIC_SHIFT, \
        f'max_shift {max_shift} exceeds realistic detector jitter cap ({MAX_REALISTIC_SHIFT}ms)'

    context_size = CONTEXT_PRE + CONTEXT_POST
    win_size     = WINDOW_PRE  + WINDOW_POST
    rng         = np.random.default_rng(seed)
    src_indices = [i for i, b in enumerate(beats)
                   if getattr(b, 'context_buffer', None) is not None
                   and getattr(b, 'pt_buffer', None) is not None]
    src_beats   = [beats[i] for i in src_indices]

    synthetic = []
    for _ in range(n):
        idb = rng.integers(len(src_beats))
        src = src_beats[idb]
        s   = int(rng.integers(-max_shift, max_shift + 1))

        beat = copy.copy(src)
        beat.parent = src_indices[idb]  # index into input beats list
        win_lo = CONTEXT_PRE + CONTEXT_SHIFT_MAX - WINDOW_PRE + s  # spike at WINDOW_PRE in slice
        beat.context_window  = src.context_buffer[:, CONTEXT_SHIFT_MAX + s :
                                                      CONTEXT_SHIFT_MAX + s + context_size]
        beat.decision_window = src.pt_buffer[:, CONTEXT_SHIFT_MAX + s :
                                               CONTEXT_SHIFT_MAX + s + win_size]
        beat.window          = src.context_buffer[:, win_lo : win_lo + win_size]
        beat.window_pre      = WINDOW_PRE - s
        beat.shift           = s   # for inspection
        synthetic.append(beat)

    return synthetic


# =========================================================
# Quick test
# =========================================================

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from beat import process_study

    beats, _, _ = process_study('data/p2_1/ecg_data.txt')

    print('--- HuBERT dataset ---')
    ds = BeatDataset(beats, transform=preprocess_hubert)
    x, d, y = ds[0]
    print(f'  x : {tuple(x.shape)}')
    print(f'  d : {tuple(d.shape)}')
    print(f'  y : {tuple(y.shape)}  qrs={y[0].sum():.0f}ms  qt={y[1].sum():.0f}ms')

    dl = DataLoader(ds, batch_size=4, shuffle=True)
    xb, db, yb = next(iter(dl))
    print(f'  batch x:{tuple(xb.shape)}  d:{tuple(db.shape)}  y:{tuple(yb.shape)}')

    # ── original beats visual check ──────────────────────────────────
    print('\n--- original beats ---')
    ann_beats = [b for b in beats if b.qrs_duration is not None][:5]
    fig, axes = plt.subplots(len(ann_beats), 2, figsize=(12, 3 * len(ann_beats)), sharex='col')
    t = np.arange(WINDOW_PRE + WINDOW_POST)
    for ax_row, beat in zip(axes, ann_beats):
        mask   = build_mask(beat)
        dec_ii = beat.decision_window[1]   # Lead II PT signal
        ax_row[0].plot(t, dec_ii, color='steelblue')
        ax_row[0].fill_between(t, 0, dec_ii.max(),
                               where=mask[0] > 0, alpha=0.3, color='red',   label='QRS')
        ax_row[0].fill_between(t, 0, dec_ii.max(),
                               where=mask[1] > 0, alpha=0.2, color='green', label='QT')
        ax_row[0].axvline(beat.window_pre, color='k', linestyle='--', linewidth=0.8)
        ax_row[0].set_title(f'spike={beat.spike_idx}ms  |  decision + mask')
        ax_row[0].legend(fontsize=7)

        lead_ii = beat.context_window[1]
        ax_row[1].plot(np.linspace(0, WINDOW_PRE + WINDOW_POST, len(lead_ii)), lead_ii,
                       color='dimgray', linewidth=0.6)
        ax_row[1].set_title(f'spike={beat.spike_idx}ms  |  context lead II')

    plt.tight_layout()
    plt.savefig('original_beats_check.png', dpi=120)
    print('  saved original_beats_check.png')
    plt.show()

    # ── amplitude scaling augmentation visual check ───────────────────
    print('\n--- amplitude scaling augmentation ---')
    ann_beats = [b for b in beats if b.qrs_duration is not None]
    expansion = generate_expansion_scale(ann_beats, n=5, seed=0)

    fig, axes = plt.subplots(len(expansion), 2, figsize=(12, 3 * len(expansion)), sharex='col')
    t = np.arange(WINDOW_PRE + WINDOW_POST)
    for ax_row, beat in zip(axes, expansion):
        mask   = build_mask(beat)
        dec_ii = beat.decision_window[1]
        ax_row[0].plot(t, dec_ii, color='steelblue')
        ax_row[0].fill_between(t, 0, dec_ii.max(),
                               where=mask[0] > 0, alpha=0.3, color='red',   label='QRS')
        ax_row[0].fill_between(t, 0, dec_ii.max(),
                               where=mask[1] > 0, alpha=0.2, color='green', label='QT')
        ax_row[0].axvline(beat.window_pre, color='k', linestyle='--', linewidth=0.8)
        ax_row[0].set_title(f'spike={beat.spike_idx}ms  |  decision + mask')
        ax_row[0].legend(fontsize=7)

        lead_ii = beat.context_window[1]
        ax_row[1].plot(np.linspace(0, WINDOW_PRE + WINDOW_POST, len(lead_ii)), lead_ii,
                       color='dimgray', linewidth=0.6)
        scales_str = '  '.join(f'L{i}:{s:.2f}' for i, s in enumerate(beat.lead_scales))
        ax_row[1].set_title(f'spike={beat.spike_idx}ms  |  scales: {scales_str}', fontsize=7)

    plt.tight_layout()
    plt.savefig('expansion_scale_check.png', dpi=120)
    print('  saved expansion_scale_check.png')
    plt.show()

    # ── shift augmentation visual check ──────────────────────────────
    print('\n--- shift augmentation ---')
    MAX_SHIFT = 30
    src_beat  = ann_beats[0]
    shifts    = [-MAX_SHIFT, 0, MAX_SHIFT]
    variants  = []
    for s in shifts:
        b = copy.copy(src_beat)
        win_size     = WINDOW_PRE + WINDOW_POST
        context_size = CONTEXT_PRE + CONTEXT_POST
        ctx_lo = CONTEXT_SHIFT_MAX + s
        win_lo = CONTEXT_PRE + CONTEXT_SHIFT_MAX - WINDOW_PRE + s  # spike at WINDOW_PRE in slice
        b.context_window  = src_beat.context_buffer[:, ctx_lo : ctx_lo + context_size]
        b.decision_window = src_beat.pt_buffer[:,     CONTEXT_SHIFT_MAX + s :
                                                        CONTEXT_SHIFT_MAX + s + win_size]
        b.window          = src_beat.context_buffer[:, win_lo : win_lo + win_size]
        b.window_pre      = WINDOW_PRE - s
        b.shift           = s
        variants.append(b)

    cmap12      = plt.cm.tab20(np.linspace(0, 0.9, 12))
    offset_step = 1.5
    t_win       = np.arange(WINDOW_PRE + WINDOW_POST)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle('Shift augmentation — all 12 leads superimposed\n'
                 'GT mask tracks the shifted spike  |  dashed = unshifted reference anchor', fontsize=10)

    for ax, beat in zip(axes, variants):
        mask = build_mask(beat)   # GT mask for THIS shifted window
        for li in range(12):
            sig = beat.window[li]
            sig_n = (sig - sig.mean()) / (sig.std() + 1e-8)
            ax.plot(t_win, sig_n + li * offset_step, color=cmap12[li], lw=0.6, alpha=0.75)

        # actual spike position in this shifted window
        ax.axvline(beat.window_pre, color='red', lw=1.2, label='spike (shifted anchor)')

        # reference: where spike would be with zero shift
        ax.axvline(WINDOW_PRE, color='black', lw=1.0, ls='--', label='reference (no shift)')

        # GT mask for this window — tracks with the shifted spike
        qrs_lo = np.argmax(mask[0] > 0)
        qrs_hi = len(mask[0]) - np.argmax(mask[0][::-1] > 0)
        ax.axvspan(qrs_lo, qrs_hi, alpha=0.10, color='red',   label='GT QRS')
        qt_lo  = np.argmax(mask[1] > 0)
        qt_hi  = len(mask[1]) - np.argmax(mask[1][::-1] > 0)
        ax.axvspan(qt_lo,  qt_hi,  alpha=0.07, color='green', label='GT QT')

        ax.set_title(f'shift={beat.shift:+d}ms', fontsize=10)
        ax.set_xlabel('sample (ms)')
        ax.set_xticks([0, WINDOW_PRE, WINDOW_PRE + WINDOW_POST])
        ax.set_xticklabels([0, f'WP={WINDOW_PRE}', WINDOW_PRE + WINDOW_POST])
        ax.set_yticks([])
        ax.grid(alpha=0.1)

    axes[0].legend(fontsize=7, loc='upper left')
    plt.tight_layout()
    plt.savefig('expansion_shift_check.png', dpi=120)
    print('  saved expansion_shift_check.png')
    plt.show()
