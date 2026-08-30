import os
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d

FS           = 1000   # Hz  (1 sample == 1 ms)
WINDOW_PRE   = 150     # samples before spike
WINDOW_POST  = 400    # samples after spike  (QRS + ST + T-wave)
WINDOW_SIZE  = WINDOW_PRE + WINDOW_POST   # 550

CONTEXT_PRE  = 2500   # samples before spike for encoder context (2.5 s at 1 kHz)
CONTEXT_POST = 2500   # samples after  spike for encoder context (2.5 s at 1 kHz)
CONTEXT_SHIFT_MAX = 500  # max shift for augmentation (samples at FS = ms)

_ANNOTATION_SECTIONS = {
    'Period Data', 'QRS Data', 'QT Data',
    'Extrasystole Data', 'Arrhythmia Data',
}


# =========================================================
# 1. BEAT  —  one spike + multi-channel window + annotations
# =========================================================

class Beat:
    def __init__(self, spike_idx, window):
        """
        spike_idx : int          sample index in the raw signal (== ms at 1 kHz)
        window    : np.ndarray   shape (n_leads, WINDOW_PRE+WINDOW_POST)
        """
        self.spike_idx    = spike_idx
        self.spike_time   = spike_idx          # ms
        self.window       = window             # (n_leads, window_size)
        self.window_pre   = WINDOW_PRE         # samples before spike in window
        self.window_post  = WINDOW_POST        # actual signal samples after spike (excl. zero-padding)
        self.paraent=-1            # index of the parent window in the original signal, used for data augmentation   -1 if original

        self.label        = None               # 0=normal 1=arrhythmia 2=extrasystole
        self.bcl          = None               # basic cycle length ms (spike[i] - spike[i-1])
        self.period       = None               # RR interval ms (from annotation)
        self.qrs_duration         = None
        self.qrs_start            = None       # absolute ms position of QRS onset
        self.qrs_start_uncert     = None       # half-width of onset transition (ms)
        self.qrs_end_uncert       = None       # half-width of offset transition (ms)
        self.qt_interval          = None
        self.qt_start             = None       # absolute ms position of QT onset
        self.noisy        = False              # True if window overlaps another beat
        self.source       = None               # filepath this beat was extracted from

        self.context_window   = None          # (n_leads, CONTEXT_PRE+CONTEXT_POST) — 5 s centred on spike
        self.spike_in_context = CONTEXT_PRE   # sample index of spike inside context_window (at FS)
        self.decision_window  = None          # (WINDOW_PRE+WINDOW_POST,) Pan-Tompkins integrated signal
        self.context_buffer   = None          # (n_leads, CONTEXT_PRE+CONTEXT_POST+2*CONTEXT_SHIFT_MAX) — wider slice for shift augmentation
        self.pt_buffer        = None          # (WINDOW_PRE+WINDOW_POST+2*CONTEXT_SHIFT_MAX,) — wider PT slice

    def annotate(self, label=None, period=None,
                 qrs_duration=None, qrs_start=None,
                 qt_interval=None, qt_start=None):
        if label        is not None: self.label        = label
        if period       is not None: self.period       = period
        if qrs_duration is not None: self.qrs_duration = qrs_duration
        if qrs_start    is not None: self.qrs_start    = qrs_start
        if qt_interval  is not None: self.qt_interval  = qt_interval
        if qt_start     is not None: self.qt_start     = qt_start

    @property
    def is_annotated(self):
        return self.label is not None

    def __repr__(self):
        return (f"Beat(spike={self.spike_idx}ms, label={self.label}, "
                f"qrs={self.qrs_duration}, qt={self.qt_interval}, "
                f"noisy={self.noisy})")


def mark_noisy_beats(beats):
    """Mark beats whose windows overlap with any other beat's window in-place.

    A beat's window spans [spike_idx - WINDOW_PRE, spike_idx + WINDOW_POST).
    Two beats are considered overlapping (noisy) when their windows intersect.
    """
    intervals = np.array(
        [(b.spike_idx - WINDOW_PRE, b.spike_idx + WINDOW_POST) for b in beats]
    )
    for i, beat in enumerate(beats):
        lo_i, hi_i = intervals[i]
        for j in range(len(beats)):
            if i == j:
                continue
            lo_j, hi_j = intervals[j]
            if lo_i < hi_j and hi_i > lo_j:   # intervals overlap
                beat.noisy = True
                break
    return beats


def recover_noisy_beats(beats, signal_matrix):
    """Re-extract windows for noisy beats, always protecting each beat's beginning.

    The right boundary of each beat is cut at the start of the next beat's
    natural window (next_spike - WINDOW_PRE), so the next beat's pre-spike
    region is always preserved intact.  The lost tail is compensated by
    extending the current beat's start leftward.  Any remaining gap (signal
    edge or extremely short RR interval) is zero-padded so the window shape
    stays constant.

    Beats whose available range is less than half the window size are left
    noisy as they cannot carry useful information.

    Parameters
    ----------
    beats         : list[Beat]   output of mark_noisy_beats (modified in-place)
    signal_matrix : np.ndarray   shape (n_leads, n_samples)

    Returns
    -------
    beats : list[Beat]
    """
    n_leads, n_samples = signal_matrix.shape
    window_size = WINDOW_PRE + WINDOW_POST

    sorted_beats = sorted(beats, key=lambda b: b.spike_idx)
    spike_pos    = [b.spike_idx for b in sorted_beats]
    n            = len(sorted_beats)

    for i, beat in enumerate(sorted_beats):
        if not beat.noisy:
            continue

        spike = beat.spike_idx

        # Right limit: start of the next beat's natural window.
        # Left limit: midpoint to the previous beat.
        lo_hard = (spike_pos[i - 1] + spike) // 2 if i > 0     else 0
        hi_hard = spike_pos[i + 1] - WINDOW_PRE   if i < n - 1 else n_samples

        hi = min(spike + WINDOW_POST, hi_hard, n_samples)
        # Extend start leftward to recover what the clipped tail lost
        lo = max(hi - window_size, lo_hard, 0)

        if hi - lo < window_size // 2:
            continue   # too cramped to be useful

        actual = signal_matrix[:, lo:hi]
        got    = hi - lo
        if got < window_size:
            padded          = np.empty((n_leads, window_size), dtype=signal_matrix.dtype)
            padded[:, :got] = actual
            padded[:, got:] = actual[:, -1:]   # extend last real sample instead of zero
            beat.window     = padded
        else:
            beat.window     = actual.copy()
        beat.window_pre  = spike - lo      # may be > WINDOW_PRE if start extended
        beat.window_post = hi - spike
        beat.noisy       = False

    return beats


def _extract_slice(signal_matrix, center, pre, post):
    """Extract a (n_leads, pre+post) slice centred at `center`, zero-padding edges."""
    n_leads, n_samples = signal_matrix.shape
    size     = pre + post
    lo       = center - pre
    hi       = center + post
    pad_left  = max(0, -lo)
    lo_clamp  = max(0, lo)
    hi_clamp  = min(n_samples, hi)
    out = np.zeros((n_leads, size), dtype=signal_matrix.dtype)
    out[:, pad_left: pad_left + (hi_clamp - lo_clamp)] = signal_matrix[:, lo_clamp:hi_clamp]
    return out


def extract_context_windows(signal_matrix, beats):
    """Attach a 5-second encoder context window to each beat, centred on its spike.

    Also stores context_buffer — a wider slice (±CONTEXT_SHIFT_MAX extra) for
    shift augmentation.  beat.spike_in_context records where the spike falls
    inside context_window (at FS).
    """
    for beat in beats:
        beat.context_window   = _extract_slice(signal_matrix, beat.spike_idx,
                                               CONTEXT_PRE, CONTEXT_POST)
        beat.context_buffer   = _extract_slice(signal_matrix, beat.spike_idx,
                                               CONTEXT_PRE  + CONTEXT_SHIFT_MAX,
                                               CONTEXT_POST + CONTEXT_SHIFT_MAX)
        beat.spike_in_context = CONTEXT_PRE  # always centred; pad shifts signal, not spike

    return beats


# =========================================================
# 2. FILE PARSER  —  all leads + embedded annotations
# =========================================================

def load_ecg(filepath):
    """Parse a study file and return all leads and annotation sections.

    Returns
    -------
    leads : dict[str, np.ndarray]
        Lead name → signal array of shape (n_samples,).
    annotations : dict[str, list[list[float]]]
        Section name → list of rows, each row a list of floats.

    Lead order in the file: VD d, I, II, III, AVR, AVL, AVF, V1..V6
    Annotation sections: Period Data, QRS Data, QT Data,
                         Extrasystole Data, Arrhythmia Data
    """
    leads       = {}
    annotations = {}

    current_name = None
    current_mode = None        # 'lead' | 'annotation'
    current_data = []
    in_curves    = False

    with open(filepath) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if not in_curves:
                if line == 'Curves:':
                    in_curves = True
                continue

            if line.endswith(':'):
                # Save previous section
                if current_name is not None:
                    if current_mode == 'lead':
                        leads[current_name] = np.array(
                            current_data, dtype=np.float32)
                    else:
                        annotations[current_name] = current_data

                section_name = line[:-1]          # strip trailing ':'
                current_name = section_name
                current_data = []
                current_mode = (
                    'annotation' if section_name in _ANNOTATION_SECTIONS
                    else 'lead'
                )
                continue

            if current_mode == 'lead':
                try:
                    current_data.append(float(line))
                except ValueError:
                    pass
            elif current_mode == 'annotation':
                try:
                    current_data.append(
                        [float(v.strip()) for v in line.split(',')])
                except ValueError:
                    pass

    # Flush last section
    if current_name is not None:
        if current_mode == 'lead':
            leads[current_name] = np.array(current_data, dtype=np.float32)
        else:
            annotations[current_name] = current_data

    return leads, annotations


def leads_matrix(leads):
    """Stack all leads into a (n_leads, n_samples) matrix in canonical order."""
    order  = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'VD d']
    names  = [n for n in order if n in leads]
    matrix = np.stack([leads[n] for n in names], axis=0)
    return matrix, names


# =========================================================
# 3. SPIKE DETECTOR  —  Pan–Tompkins
# =========================================================

def _pan_tompkins_core(
    sig,
    fs,
    min_distance_ms=150,
    band=(5, 40),
    integration_window=90,
    percentile=80,
):
    """
    Shared Pan–Tompkins preprocessing + peak detection.

    Returns
    -------
    filtered   : bandpassed ECG
    integrated : decision variable
    peaks      : detected peaks (DELAYED)
    thr2       : final adaptive threshold
    delay      : integration delay (samples)
    """
    sig = sig.astype(np.float64)

    # 1. Bandpass filter
    b, a = butter(2, band, btype="band", fs=fs)
    filtered = filtfilt(b, a, sig)

    # 2. Derivative
    dy = np.diff(filtered, prepend=filtered[0])

    # 3–4. Squaring + moving window integration
    integrated = uniform_filter1d(dy**2, size=integration_window)
    integrated = np.clip(integrated, a_min=0, a_max=120)  # numerical stability
    delay = integration_window // 2

    # 5. Adaptive thresholding
    thr1 = np.percentile(integrated, percentile)

    peaks, props = find_peaks(
        integrated,
        height=thr1,
        distance=int(min_distance_ms),
    )

    if len(peaks) > 0:
        heights = props["peak_heights"]
        thr2 = 0.5 * heights.mean()

        peaks, _ = find_peaks(
            integrated,
            height=thr2,
            distance=min_distance_ms,
        )
    else:
        thr2 = thr1

    return filtered, integrated, peaks, thr2, delay


def _pan_tompkins_detect(leads, min_distance_ms=1000, thr2_factor=0.40):
    """Multi-lead Pan–Tompkins — returns all intermediate signals.

    Builds a combined decision variable by z-score averaging the PT energy
    across all PT_LEADS present in `leads`, then detects and refines peaks.

    Returns
    -------
    combined  : 1-D array     z-score-averaged PT energy across PT_LEADS
    delay     : int           integration delay (samples)
    filt_ref  : 1-D array     bandpass-filtered reference lead (II or best available)
    peaks     : 1-D int array peaks in combined signal space (before delay correction)
    thr2      : float         adaptive threshold used for peaks
    refined   : list[int]     peaks snapped to local R-peak in filt_ref (delay-corrected)
    """
    available = [n for n in PT_LEADS if n in leads]
    combined = None
    delay    = None
    n_used   = 0
    for name in available:
        _, integrated, _, _, d = _pan_tompkins_core(
            leads[name], FS, min_distance_ms=min_distance_ms,
        )
        mu, sd = integrated.mean(), integrated.std()
        if sd < 1e-9:
            continue
        normed   = (integrated - mu) / sd
        combined = normed if combined is None else combined + normed
        delay    = d
        n_used  += 1

    if combined is None:
        # fallback: single lead
        sig = leads.get("II", next(iter(leads.values())))
        _, combined, _, _, delay = _pan_tompkins_core(sig, FS, min_distance_ms=min_distance_ms)
    else:
        combined = combined / n_used

    thr = np.percentile(combined, 80)
    peaks, props = find_peaks(combined, height=thr, distance=int(min_distance_ms))
    if len(peaks) > 0:
        thr2 = thr2_factor * props["peak_heights"].mean()
        peaks, _ = find_peaks(combined, height=thr2, distance=min_distance_ms)
    else:
        thr2 = thr

    ref_lead = leads.get("II", leads.get("V5", next(iter(leads.values()))))
    b, a = butter(2, (5, 40), btype="band", fs=FS)
    filt_ref = filtfilt(b, a, ref_lead.astype(np.float64))

    refined = []
    for p in peaks:
        p_corr = p - delay
        lo = max(0, p_corr - 20)
        hi = min(len(filt_ref), p_corr + 21)
        # use absolute value so inverted leads (e.g. AVR) don't break refinement
        idx = np.argmax(np.abs(filt_ref[lo:hi]))
        refined.append(lo + idx)

    return combined, delay, filt_ref, peaks, thr2, refined


def detect_spikes(leads, min_distance_ms=100, thr2_factor=0.40):
    """Detect R-peaks using combined multi-lead Pan–Tompkins.

    Calls _pan_tompkins_detect then deduplicates refined peaks that ended up
    within min_distance_ms of each other, keeping the highest-amplitude one.
    """
    combined, delay, filt_ref, peaks, thr2, refined = _pan_tompkins_detect(
        leads, min_distance_ms, thr2_factor=thr2_factor
    )

    # post-refinement deduplication
    refined = sorted(refined)
    amps    = [abs(filt_ref[r]) for r in refined]
    keep    = []
    i = 0
    while i < len(refined):
        j = i + 1
        while j < len(refined) and refined[j] - refined[i] < min_distance_ms:
            j += 1
        best = max(range(i, j), key=lambda k: amps[k])
        keep.append(refined[best])
        i = j

    return np.array(keep, dtype=int)


def pan_tompkins_signal(leads):
    """Return decision variable + threshold for plotting."""
    sig = leads["II"]
    _, integrated, _, thr2, _ = _pan_tompkins_core(sig, FS)
    return integrated, thr2


# =========================================================
# 4. WINDOW EXTRACTION  —  multi-channel
# =========================================================

def extract_windows(signal_matrix, spike_indices):
    """Slice an asymmetric window around each spike across all leads.
    Window spans [spike - WINDOW_PRE, spike + WINDOW_POST).

    Parameters
    ----------
    signal_matrix : np.ndarray  shape (n_leads, n_samples)
    spike_indices : np.ndarray  1-D array of spike sample positions

    Returns
    -------
    list[Beat]  — only spikes with a full window inside the signal are kept
    """
    n_samples = signal_matrix.shape[1]
    beats = []
    for idx in spike_indices:
        lo = idx - WINDOW_PRE
        hi = idx + WINDOW_POST
        if lo < 0 or hi > n_samples:
            continue
        window = signal_matrix[:, lo:hi].copy()   # (n_leads, window_size)
        beats.append(Beat(spike_idx=int(idx), window=window))
    return beats


# =========================================================
# 5. ANNOTATION  —  match spikes to marked beats
# =========================================================

def annotate_beats(beats, annotations, tol_ms=150.0):
    """Attach QRS / QT / label to each beat using embedded annotations.

    Matching strategy
    -----------------
    QRS Data row : [qrs_start_ms, qrs_end_ms, period_ms, qrs_duration_ms]
        Spike matches if it falls within [qrs_start - tol_ms, qrs_end + tol_ms].

    QT Data row  : [qt_start_ms, qt_end_ms, period_ms, qt_duration_ms]
        Same window-based matching.

    Label (0=normal / 1=arrhythmia / 2=extrasystole)
        Arrhythmia / Extrasystole rows: [period, duration, t_start, t_end]
        A beat is labelled if its spike_time falls in [t_start, t_end].
        Unmatched beats receive label 0 (normal).
    """
    qrs_data = np.array(annotations.get('QRS Data', []))
    qt_data  = np.array(annotations.get('QT Data',  []))

    arr_rows = annotations.get('Arrhythmia Data',   [])
    ext_rows = annotations.get('Extrasystole Data', [])
    arr_wins = [(r[0], r[1]) for r in arr_rows if len(r) >= 4]
    ext_wins = [(r[0], r[1]) for r in ext_rows if len(r) >= 4]

    # ── label assignment (per-beat, no conflict possible) ────────────
    for beat in beats:
        t = beat.spike_time
        if any(t_s <= t <= t_e for t_s, t_e in arr_wins):
            beat.annotate(label=1)
        elif any(t_s <= t <= t_e for t_s, t_e in ext_wins):
            beat.annotate(label=2)
        else:
            beat.annotate(label=0)

    # ── QRS / QT: global one-to-one assignment ────────────────────────
    # Each annotation row goes to the single closest spike; each spike
    # gets at most one annotation.  Ties broken by distance to interval.
    def _assign(data, apply_fn):
        if not len(data):
            return
        candidates = []
        for bi, beat in enumerate(beats):
            t = beat.spike_time
            for ri, row in enumerate(data):
                start, end = row[0], row[1]
                if start - tol_ms <= t <= end + tol_ms:
                    dist = abs(t - (start + end) / 2)  # closest spike to QRS midpoint wins
                    candidates.append((dist, bi, ri))
        candidates.sort()
        used_beats, used_rows = set(), set()
        for dist, bi, ri in candidates:
            if bi in used_beats or ri in used_rows:
                continue
            apply_fn(beats[bi], data[ri])
            used_beats.add(bi)
            used_rows.add(ri)

    _assign(qrs_data, lambda b, row: b.annotate(
        period=float(row[2]), qrs_duration=float(row[3]), qrs_start=float(row[0])))
    _assign(qt_data,  lambda b, row: b.annotate(
        qt_interval=float(row[3]), qt_start=float(row[0])))

    return beats


# =========================================================
# 6. FULL PIPELINE
# =========================================================

# Standard 12-lead ECG leads used for PT energy (excludes 'VD d' stimulus electrode)
PT_LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
PT_N_LEADS = len(PT_LEADS)  # 12


def extract_decision_windows(leads, lead_names, beats):
    """Attach the Pan-Tompkins integrated signal window to each beat.

    The window spans [spike_idx - WINDOW_PRE, spike_idx + WINDOW_POST), matching
    the shape of beat.window.  Regions outside the signal are zero-padded.
    Also stores pt_buffer — a wider slice (±CONTEXT_SHIFT_MAX extra) for
    shift augmentation.

    decision_window : (PT_N_LEADS, win_size)   — PT energy per standard lead
    pt_buffer       : (PT_N_LEADS, buf_size)   — wider slice for shift augmentation
    """
    pt_names = [n for n in PT_LEADS if n in leads]
    integrated_signals = []
    for name in pt_names:
        _, integ, _, _, _ = _pan_tompkins_core(leads[name], FS)
        integrated_signals.append(integ)
    n_leads  = len(integrated_signals)
    n        = len(integrated_signals[0])
    win_size = WINDOW_PRE + WINDOW_POST
    buf_size = win_size + 2 * CONTEXT_SHIFT_MAX

    for beat in beats:
        for pre, post, size, attr in [
            (WINDOW_PRE,                     WINDOW_POST,                     win_size, 'decision_window'),
            (WINDOW_PRE + CONTEXT_SHIFT_MAX, WINDOW_POST + CONTEXT_SHIFT_MAX, buf_size, 'pt_buffer'),
        ]:
            lo   = beat.spike_idx - pre
            hi   = beat.spike_idx + post
            lo_c = max(0, lo)
            hi_c = min(n, hi)
            w    = np.zeros((n_leads, size), dtype=np.float32)
            for li, integrated in enumerate(integrated_signals):
                w[li, lo_c - lo : hi_c - lo] = integrated[lo_c : hi_c]
            setattr(beat, attr, w)

    return beats


def compute_bcl(beats):
    """Set beat.bcl = spike[i] - spike[i-1] for all beats (in spike order).

    Beats are assumed to come from the same study and be sorted by spike_idx.
    The first beat gets bcl=None (no predecessor).
    """
    for i, beat in enumerate(beats):
        beat.bcl = (beat.spike_idx - beats[i - 1].spike_idx) if i > 0 else None


def process_study(filepath):
    """Load one study file and return annotated Beat objects + lead names.

    Returns
    -------
    beats      : list[Beat]
    lead_names : list[str]
    leads      : dict[str, np.ndarray]
    """
    leads, annotations = load_ecg(filepath)
    matrix, lead_names = leads_matrix(leads)
    spikes             = detect_spikes(leads)
    beats              = extract_windows(matrix, spikes)
    annotate_beats(beats, annotations)
    compute_bcl(beats)
    extract_context_windows(matrix, beats)
    extract_decision_windows(leads, lead_names, beats)
    for beat in beats:
        beat.source = filepath
    return beats, lead_names, leads


# =========================================================
# 7. DATASET UTILITIES
# =========================================================

_BEAT_CACHE_VERSION = 8   # bump when Beat fields or save format change
_FLOAT_FIELDS = ('bcl', 'period', 'qrs_duration', 'qrs_start', 'qt_interval', 'qt_start')


def _beats_npy_dir(cache_dir, key):
    return os.path.join(cache_dir, f'beats_{key}')


def _save_beats_npy(result, npy_dir):
    """Serialize (annotated, unannotated, noisy) as stacked npy files."""
    import json
    os.makedirs(npy_dir, exist_ok=True)

    for split, beats in zip(('ann', 'unann', 'noisy'), result):
        N = len(beats)
        np.save(os.path.join(npy_dir, f'{split}_n.npy'), np.array([N], dtype=np.int32))
        if N == 0:
            continue

        np.save(os.path.join(npy_dir, f'{split}_window.npy'),
                np.stack([b.window for b in beats]).astype(np.float32))
        np.save(os.path.join(npy_dir, f'{split}_spike_idx.npy'),
                np.array([b.spike_idx  for b in beats], dtype=np.int32))
        np.save(os.path.join(npy_dir, f'{split}_window_pre.npy'),
                np.array([b.window_pre for b in beats], dtype=np.int32))
        np.save(os.path.join(npy_dir, f'{split}_label.npy'),
                np.array([b.label if b.label is not None else -1
                          for b in beats], dtype=np.int16))
        np.save(os.path.join(npy_dir, f'{split}_noisy.npy'),
                np.array([b.noisy for b in beats], dtype=bool))
        for field in _FLOAT_FIELDS:
            np.save(os.path.join(npy_dir, f'{split}_{field}.npy'),
                    np.array([getattr(b, field) if getattr(b, field) is not None else np.nan
                              for b in beats], dtype=np.float32))
        with open(os.path.join(npy_dir, f'{split}_source.json'), 'w') as fh:
            json.dump([b.source for b in beats], fh)

        if getattr(beats[0], 'context_window', None) is not None:
            np.save(os.path.join(npy_dir, f'{split}_context_window.npy'),
                    np.stack([b.context_window  for b in beats]).astype(np.float32))
            np.save(os.path.join(npy_dir, f'{split}_context_buffer.npy'),
                    np.stack([b.context_buffer  for b in beats]).astype(np.float32))
            np.save(os.path.join(npy_dir, f'{split}_decision_window.npy'),
                    np.stack([b.decision_window for b in beats]).astype(np.float32))
            np.save(os.path.join(npy_dir, f'{split}_pt_buffer.npy'),
                    np.stack([b.pt_buffer       for b in beats]).astype(np.float32))

    np.save(os.path.join(npy_dir, 'version.npy'),
            np.array([_BEAT_CACHE_VERSION], dtype=np.int32))


def _load_beats_npy(npy_dir):
    """Deserialize (annotated, unannotated, noisy) from stacked npy files.

    All arrays are read fully into RAM upfront (no mmap).
    """
    import json
    result = []
    for split in ('ann', 'unann', 'noisy'):
        N = int(np.load(os.path.join(npy_dir, f'{split}_n.npy'))[0])
        if N == 0:
            result.append([])
            continue

        window      = np.load(os.path.join(npy_dir, f'{split}_window.npy'))
        spike_idx   = np.load(os.path.join(npy_dir, f'{split}_spike_idx.npy'))
        window_pre  = np.load(os.path.join(npy_dir, f'{split}_window_pre.npy'))
        label       = np.load(os.path.join(npy_dir, f'{split}_label.npy'))
        noisy_arr   = np.load(os.path.join(npy_dir, f'{split}_noisy.npy'))
        float_arrs  = {f: np.load(os.path.join(npy_dir, f'{split}_{f}.npy'))
                       for f in _FLOAT_FIELDS}
        with open(os.path.join(npy_dir, f'{split}_source.json')) as fh:
            sources = json.load(fh)

        ctx_path = os.path.join(npy_dir, f'{split}_context_window.npy')
        if os.path.exists(ctx_path):
            context_window  = np.load(ctx_path)
            context_buffer  = np.load(os.path.join(npy_dir, f'{split}_context_buffer.npy'))
            decision_window = np.load(os.path.join(npy_dir, f'{split}_decision_window.npy'))
            pt_buffer       = np.load(os.path.join(npy_dir, f'{split}_pt_buffer.npy'))
        else:
            context_window = context_buffer = decision_window = pt_buffer = None

        beats = []
        for i in range(N):
            beat            = Beat(int(spike_idx[i]), window[i])
            beat.window_pre = int(window_pre[i])
            beat.label      = None if label[i] == -1 else int(label[i])
            beat.noisy      = bool(noisy_arr[i])
            beat.source     = sources[i]
            for field, arr in float_arrs.items():
                v = float(arr[i])
                setattr(beat, field, None if np.isnan(v) else v)
            if context_window is not None:
                beat.context_window  = context_window[i]
                beat.context_buffer  = context_buffer[i]
                beat.decision_window = decision_window[i]
                beat.pt_buffer       = pt_buffer[i]
            beats.append(beat)

        result.append(beats)
    return tuple(result)


def load_or_process_beats(folders, cache_dir='cache', force=False,
                          ecg_filename='ecg_data.txt'):
    """Return (annotated, unannotated, noisy), loading from npy cache when available.

    Cache key: sorted folder basenames.  Use force=True to reprocess and overwrite.
    All arrays are loaded fully into RAM on read (no lazy mmap).
    """
    key     = '_'.join(sorted(os.path.basename(f) for f in folders))
    npy_dir = _beats_npy_dir(cache_dir, key)
    ver_path = os.path.join(npy_dir, 'version.npy')

    cache_ok = (not force
                and os.path.isdir(npy_dir)
                and os.path.exists(ver_path)
                and int(np.load(ver_path)[0]) == _BEAT_CACHE_VERSION)

    if cache_ok:
        print(f'  [cache] loading beats from {npy_dir}/')
        return _load_beats_npy(npy_dir)

    result = load_patient_beats(folders, ecg_filename)
    os.makedirs(cache_dir, exist_ok=True)
    _save_beats_npy(result, npy_dir)
    print(f'  [cache] saved beats to  {npy_dir}/')
    return result


def _process_folder(args):
    """Worker: process one folder and return (annotated, unannotated, noisy) lists."""
    folder, ecg_filename = args
    filepath = os.path.join(folder, ecg_filename)
    if not os.path.isfile(filepath):
        return [], [], []
    try:
        beats, _, _ = process_study(filepath)
    except Exception:
        return [], [], []

    annotated, unannotated, noisy = [], [], []
    for beat in beats:
        if beat.noisy:
            noisy.append(beat)
        elif beat.qrs_duration is not None and beat.qt_interval is not None:
            annotated.append(beat)
        else:
            unannotated.append(beat)
    return annotated, unannotated, noisy


def load_patient_beats(folders, ecg_filename="ecg_data.txt"):
    """Process a list of patient folders and partition beats by quality.

    Parameters
    ----------
    folders      : list[str]  paths to patient directories
    ecg_filename : str        name of the ECG file inside each folder

    Returns
    -------
    annotated   : list[Beat]  beats with qrs_duration and qt_interval filled
    unannotated : list[Beat]  beats missing qrs/qt annotation
    noisy       : list[Beat]  beats whose window overlaps another beat's window
                              (these may also appear in annotated/unannotated)
    """
    from concurrent.futures import ProcessPoolExecutor
    import os as _os

    annotated   = []
    unannotated = []
    noisy       = []

    args = [(f, ecg_filename) for f in folders]
    with ProcessPoolExecutor(max_workers=_os.cpu_count()) as pool:
        for ann, unann, noisy_f in pool.map(_process_folder, args):
            annotated.extend(ann)
            unannotated.extend(unann)
            noisy.extend(noisy_f)

    return annotated, unannotated, noisy


def beats_to_xy(beats):
    """Convert a list of beats into model-ready arrays.

    Parameters
    ----------
    beats : list[Beat]

    Returns
    -------
    X     : np.ndarray  shape (n_beats, n_leads, window_size)  float32
    y_qrs : np.ndarray  shape (n_beats,)  float32, NaN where annotation missing
    y_qt  : np.ndarray  shape (n_beats,)  float32, NaN where annotation missing
    """
    X     = np.stack([b.window for b in beats], axis=0).astype(np.float32)
    y_qrs = np.array(
        [b.qrs_duration if b.qrs_duration is not None else np.nan
         for b in beats],
        dtype=np.float32,
    )
    y_qt  = np.array(
        [b.qt_interval if b.qt_interval is not None else np.nan
         for b in beats],
        dtype=np.float32,
    )
    return X, y_qrs, y_qt


# =========================================================
# 8. MAIN  —  validation + feature tests
# =========================================================

def _sep(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


if __name__ == "__main__":
    import glob as _glob
    import matplotlib.pyplot as plt
    from plot import (plot_signal_windows, plot_beat, plot_beats,
                      plot_annotated_beat, plot_annotated_beats)

    DATA_DIR = "data"

    # ----------------------------------------------------------
    # C. load_patient_beats across all available folders
    # ----------------------------------------------------------
    _sep("C. load_patient_beats")

    folders = sorted(_glob.glob(os.path.join(DATA_DIR, "p7*")))
    print(f"Patient folders found: {folders}")

    annotated, unannotated, noisy = load_patient_beats(folders)

    total = len(annotated) + len(unannotated) + len(noisy)
    print(f"Total beats loaded : {total}")
    print(f"  annotated        : {len(annotated)}")
    print(f"  unannotated      : {len(unannotated)}")
    print(f"  noisy            : {len(noisy)}")

    if annotated:
        print(f"\nSample annotated beat  : {annotated[0]}")
    if unannotated:
        print(f"Sample unannotated beat: {unannotated[0]}")
    if noisy:
        print(f"Sample noisy beat      : {noisy[0]}")

    # ----------------------------------------------------------
    # D. beats_to_xy
    # ----------------------------------------------------------
    _sep("D. beats_to_xy")

    if annotated:
        X, y_qrs, y_qt = beats_to_xy(annotated)
        print(f"annotated set  →  X:{X.shape}  y_qrs:{y_qrs.shape}  y_qt:{y_qt.shape}")
        print(f"  X dtype      : {X.dtype}")
        print(f"  y_qrs range  : [{np.nanmin(y_qrs):.1f}, {np.nanmax(y_qrs):.1f}] ms")
        print(f"  y_qt  range  : [{np.nanmin(y_qt):.1f},  {np.nanmax(y_qt):.1f}] ms")
        nan_qrs = int(np.isnan(y_qrs).sum())
        nan_qt  = int(np.isnan(y_qt).sum())
        print(f"  NaN y_qrs    : {nan_qrs}  |  NaN y_qt : {nan_qt}")

    if unannotated:
        X_u, y_qrs_u, y_qt_u = beats_to_xy(unannotated)
        print(f"unannotated set → X:{X_u.shape}  "
              f"NaN qrs:{np.isnan(y_qrs_u).sum()}  "
              f"NaN qt:{np.isnan(y_qt_u).sum()}")

    # ----------------------------------------------------------
    # E. Plots  (use first annotated beat's study for signal plots)
    # ----------------------------------------------------------
    _sep("E. Plots")

    plot_file = annotated[0].spike_idx if annotated else None
    _beats_plot, lead_names, leads = process_study(folders[0] + '/ecg_data.txt')

    sig_i = leads['I']
    decision, thr = pan_tompkins_signal(leads)
    plot_signal_windows(sig_i, _beats_plot, t_start=25000-1000, t_end=25000+1000,
                        decision=decision, threshold=thr)
    plt.savefig("signal_windows.png", dpi=300, bbox_inches="tight")
    print("Saved signal_windows.png")

    plot_beats(_beats_plot[:9], lead_names)
    plt.savefig("beats.png", dpi=300, bbox_inches="tight")
    print("Saved beats.png")

    # ----------------------------------------------------------
    # F. Annotated beat plots
    # ----------------------------------------------------------
    _sep("F. Annotated beat plots")

    ann_beats = [b for b in _beats_plot if b.qrs_duration is not None]
    print(f"Beats with QRS annotation: {len(ann_beats)}")

    if ann_beats:
        plot_annotated_beat(ann_beats[0], lead_names)
        plt.savefig("annotated_beat.png", dpi=300, bbox_inches="tight")
        print("Saved annotated_beat.png")

    # ----------------------------------------------------------
    # G. Mask-vs-annotation audit
    # ----------------------------------------------------------
    _sep("G. Mask-vs-annotation audit")

    from dataset import build_mask, WINDOW_SIZE as _WS

    TOL = 1.0   # ms tolerance for rounding

    zero_qrs, zero_qt, mismatch_qrs, mismatch_qt = [], [], [], []
    for b in annotated:
        m        = build_mask(b)
        qrs_sum  = float(m[0].sum())
        qt_sum   = float(m[1].sum())
        qrs_err  = abs(qrs_sum - b.qrs_duration)
        qt_err   = abs(qt_sum  - b.qt_interval)

        ws     = b.spike_idx - b.window_pre
        qrs_lo = int(round(b.qrs_start)) - ws
        qrs_hi = qrs_lo + int(round(b.qrs_duration))
        qt_lo  = int(round(b.qt_start))  - ws
        qt_hi  = qt_lo  + int(round(b.qt_interval))

        flags = []
        if qrs_sum == 0:
            flags.append('QRS=ZERO')
            zero_qrs.append(b)
        elif qrs_err > TOL:
            flags.append(f'QRS_MISMATCH({qrs_sum:.0f}vs{b.qrs_duration:.1f})')
            mismatch_qrs.append(b)
       # if qt_sum == 0:
      #      flags.append('QT=ZERO')
       #     zero_qt.append(b)
     #   elif qt_err > TOL:
        #    flags.append(f'QT_MISMATCH({qt_sum:.0f}vs{b.qt_interval:.1f})')
        #    mismatch_qt.append(b)

        if flags:
            clip_sides = []
            if qrs_lo < 0:              clip_sides.append('QRS:start')
            if qrs_hi > _WS:            clip_sides.append('QRS:end')
            if qt_lo  < 0:              clip_sides.append('QT:start')
            if qt_hi  > _WS:            clip_sides.append('QT:end')
            src = os.path.basename(os.path.dirname(b.source)) if b.source else '?'
            print(
                f"  [{src}] spike={b.spike_idx:6d}  win=[{ws},{ws+_WS})  "
                f"qrs=[{qrs_lo},{qrs_hi}) → {qrs_sum:.0f}/{b.qrs_duration:.1f}ms  "
                f"qt=[{qt_lo},{qt_hi}) → {qt_sum:.0f}/{b.qt_interval:.1f}ms  "
                f"clip={','.join(clip_sides) or 'rounding'}  !! {' | '.join(flags)}"
            )

    clip_qrs_start = sum(1 for b in annotated
                         if (int(round(b.qrs_start)) - (b.spike_idx - b.window_pre)) < 0)
    clip_qrs_end   = sum(1 for b in annotated
                         if (int(round(b.qrs_start)) - (b.spike_idx - b.window_pre)
                             + int(round(b.qrs_duration))) > _WS)
    clip_qt_start  = sum(1 for b in annotated
                         if (int(round(b.qt_start))  - (b.spike_idx - b.window_pre)) < 0)
    clip_qt_end    = sum(1 for b in annotated
                         if (int(round(b.qt_start))  - (b.spike_idx - b.window_pre)
                             + int(round(b.qt_interval))) > _WS)

    print(f"\nSummary: {len(annotated)} annotated beats")
    print(f"  zero QRS mask      : {len(zero_qrs)}")
    print(f"  zero QT  mask      : {len(zero_qt)}")
    print(f"  QRS sum != annot   : {len(mismatch_qrs)}  (clips at start={clip_qrs_start}, end={clip_qrs_end})")
    print(f"  QT  sum != annot   : {len(mismatch_qt)}  (clips at start={clip_qt_start}, end={clip_qt_end})")
    if not any([zero_qrs, zero_qt, mismatch_qrs, mismatch_qt]):
        print("  all masks match annotations exactly")

    # ----------------------------------------------------------
    # H. Duplicate spike + wrong-annotation audit
    # ----------------------------------------------------------
    _sep("H. Duplicate spike + wrong-annotation audit")

    from collections import Counter
    # Group by source file — spike_idx is file-local so duplicates only meaningful within a file
    from itertools import groupby
    beats_by_source = {}
    for b in annotated:
        beats_by_source.setdefault(b.source, []).append(b)

    total_spike_dups = 0
    total_annot_dups = 0

    for src, src_beats in sorted(beats_by_source.items()):
        src_label = os.path.basename(os.path.dirname(src))

        # duplicate spike_idx within same study
        spike_counts = Counter(b.spike_idx for b in src_beats)
        for spike, count in sorted(spike_counts.items()):
            if count > 1:
                total_spike_dups += 1
                print(f"  [{src_label}] DUPLICATE SPIKE spike={spike}  (x{count})")

        # same qrs_start claimed by more than one beat
        qrs_counts = Counter(b.qrs_start for b in src_beats)
        for qrs_start, count in sorted(qrs_counts.items()):
            if count > 1:
                total_annot_dups += 1
                spikes = ', '.join(str(b.spike_idx)
                                   for b in src_beats if b.qrs_start == qrs_start)
                print(f"  [{src_label}] SHARED ANNOTATION qrs_start={qrs_start:.0f}  "
                      f"claimed by spikes [{spikes}]  (x{count})")

    if total_spike_dups == 0 and total_annot_dups == 0:
        print("No duplicate spikes or shared annotations found.")

    print(f"\nSummary: {total_spike_dups} duplicate spike(s), "
          f"{total_annot_dups} shared annotation(s)")

      