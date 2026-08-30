"""Recording — wrapper around a list of Beat objects for one ECG study."""

import os
import numpy as np

from scipy.signal import find_peaks as _find_peaks

from .beat import (
    leads_matrix, detect_spikes, extract_windows,
    extract_context_windows, extract_decision_windows,
    annotate_beats, compute_bcl,
    mark_noisy_beats, recover_noisy_beats,
)

# How to handle beats whose window overlaps a neighbor's (mark_noisy_beats):
#   'recovery' -- rescue with recover_noisy_beats() (asymmetric, truncated-tail
#                 window; the original/training-time behavior). Beats it can't
#                 rescue stay noisy. Beats it does rescue still get discarded
#                 (re-flagged noisy) post-prediction if qrs_duration < 100ms --
#                 a quality gate, since a short duration on a recovered window
#                 is a common symptom of a still-bad rescue.
#   'exclude'  -- no recovery attempt; skip inference entirely for beats still
#                 flagged noisy after mark_noisy_beats.
#   'force'    -- no recovery attempt; every beat keeps its plain, standard
#                 R-peak-centered window and gets a prediction regardless of
#                 the noisy flag; callers should include noisy beats in output
#                 too (ecg_marker.py's automatic_period_marking() does this by
#                 checking the mode). Beats under the duration quality check
#                 stay in the output but get their uncertainty forced to
#                 FORCE_MODE_LOW_CONFIDENCE_UNCERTAINTY instead of being
#                 discarded -- 'force' means visible, not trusted.
NOISY_BEAT_MODES = ('recovery', 'exclude', 'force')
NOISY_BEAT_MIN_DURATION_MS = 100.0  # was_noisy + short-duration quality check
FORCE_MODE_LOW_CONFIDENCE_UNCERTAINTY = 40.0  # ms, 'force' mode's flag value


def _detect_spikes_from_stimulus(vd_d, min_distance_ms=50):
    """Detect beat positions from the VD d stimulus channel.

    The pacing spike is a sharp high-amplitude artifact — much more reliable
    than Pan-Tompkins for paced rhythms in EP studies.
    """
    sig = np.abs(vd_d.astype(np.float32))
    thr = np.percentile(sig, 99) * 0.5
    peaks, _ = _find_peaks(sig, height=thr, distance=int(min_distance_ms))
    return peaks.astype(int)


class Recording:
    """All detected beats from one ECG recording.

    Attributes
    ----------
    beats        : list[Beat]   all beats in spike-time order
    lead_names   : list[str]    canonical lead order used for signal_matrix
    signal_matrix: np.ndarray   (n_leads, n_samples) float32
    """

    def __init__(self, beats, lead_names, signal_matrix):
        self._beats        = beats
        self.lead_names    = lead_names
        self.signal_matrix = signal_matrix

    # ── constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_signal(cls, leads, annotations=None, source=None, predict=False,
                     progress_callback=None, noisy_beat_mode='recovery'):
        """Build a Recording from a leads dict already in memory.

        Parameters
        ----------
        leads       : dict[str, np.ndarray]   lead name → 1-D signal at 1 kHz
        annotations : dict | None             ECG_MARKER annotation dicts;
                                              if None all beats come back unannotated
        source      : str | None              filepath label attached to each Beat
        predict     : bool                    run NN inference to fill qrs/qt fields
        progress_callback : callable | None   called as progress_callback(i, n)
                                              after each of n beats during
                                              inference (i is 1-based); for a
                                              GUI progress bar. ecg_nn itself
                                              has no UI dependency -- pass a
                                              tqdm wrapper here for a console
                                              bar, or a Tk widget updater.
        noisy_beat_mode : str                 one of NOISY_BEAT_MODES -- see
                                              module docstring above.

        Returns
        -------
        Recording
        """
        if noisy_beat_mode not in NOISY_BEAT_MODES:
            raise ValueError(f"noisy_beat_mode must be one of {NOISY_BEAT_MODES}, "
                              f"got {noisy_beat_mode!r}")

        annotations = annotations or {}
        matrix, lead_names = leads_matrix(leads)

        if 'VD d' in leads:
            spikes = _detect_spikes_from_stimulus(leads['VD d'])
        else:
            spikes = detect_spikes(leads)
        beats  = extract_windows(matrix, spikes)
        mark_noisy_beats(beats)
        if noisy_beat_mode == 'recovery':
            recover_noisy_beats(beats, matrix)
        # 'exclude' and 'force' both skip recovery: every beat keeps its
        # plain, standard R-peak-centered window from extract_windows()
        # above. They differ only in whether _run_inference computes a
        # prediction for beats still flagged noisy -- see _run_inference.
        annotate_beats(beats, annotations)
        compute_bcl(beats)
        extract_context_windows(matrix, beats)
        extract_decision_windows(leads, lead_names, beats)

        if source is not None:
            for b in beats:
                b.source = source

        rec = cls(beats, lead_names, matrix)
        rec.noisy_beat_mode = noisy_beat_mode

        if predict:
            rec._run_inference(progress_callback=progress_callback)

        return rec

    # ── inference ────────────────────────────────────────────────────────────

    _V6_CHECKPOINT = os.path.join(
        os.path.dirname(__file__), '..', 'v6_daint', 'model', 'pre_s2_ep1000.pt')

    # Model + encoder are expensive to build (HuBERT load, checkpoint read,
    # CUDA init) and don't depend on the recording, so cache them at class
    # level across calls/instances instead of rebuilding on every
    # from_signal(..., predict=True). Cleared automatically if a different
    # process starts (module-level state, not persisted).
    _model_cache = {}

    def _run_inference(self, progress_callback=None):
        """Run mask-head inference and write qrs_duration / qt_interval back
        into each beat.  Skips noisy beats.

        Model selection: v6 (MaskHeadV6, attention-pooling) if its ported
        checkpoint is present alongside the package, else the original FiLM
        MaskHead. Both consume the same (window, context) pair from the same
        HuBERT encoder, so only the head and its output unpacking differ.

        progress_callback, if given, is called as progress_callback(stage, i, n)
        where stage is 'loading_model', 'loading_encoder', or 'inference'
        (i/n are 0/0 for the loading stages, 1-based beat count for inference).
        """
        import torch
        from .model import build_model, build_model_v6, build_encoder
        from .dataset import preprocess_hubert
        from .inference import mask_to_interval

        def _notify(stage, i=0, n=0):
            if progress_callback is not None:
                progress_callback(stage, i, n)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cache_key = str(device)

        if cache_key in self._model_cache:
            head, model_version, encoder = self._model_cache[cache_key]
        else:
            _notify('loading_model')
            if os.path.exists(self._V6_CHECKPOINT):
                head, _ = build_model_v6(checkpoint_path=self._V6_CHECKPOINT, device=device)
                model_version = 'v6'
            else:
                weights_path = os.path.join(os.path.dirname(__file__), 'weights', 'film_head.pt')
                if not os.path.exists(weights_path):
                    raise FileNotFoundError(
                        f"No trained weights at {weights_path} and no v6 checkpoint at "
                        f"{self._V6_CHECKPOINT}. Train with train_film.py then re-run export.sh."
                    )
                head, _ = build_model(device=device)
                head.load_state_dict(torch.load(weights_path, map_location=device))
                model_version = 'film'
            head.eval()

            _notify('loading_encoder')
            encoder, _ = build_encoder(device=device, freeze=True)
            encoder.eval()

            self._model_cache[cache_key] = (head, model_version, encoder)

        mode = getattr(self, 'noisy_beat_mode', 'recovery')
        if mode == 'exclude':
            # No recovery attempted, and no prediction for beats still
            # flagged noisy -- they simply keep qrs_duration=None.
            targets = [b for b in self._beats if not b.noisy and b.context_window is not None]
        else:
            # 'recovery': noisy beats were already resolved to either
            # recovered (noisy=False, gets a prediction) or unrecoverable
            # (still noisy -- skip, same as 'exclude' for those).
            # 'force': every beat gets a prediction regardless of noisy.
            if mode == 'recovery':
                targets = [b for b in self._beats if not b.noisy and b.context_window is not None]
            else:
                targets = [b for b in self._beats if b.context_window is not None]
        if not targets:
            return

        from tqdm import tqdm

        n_targets = len(targets)
        _notify('inference', 0, n_targets)
        with torch.no_grad():
            for i, beat in enumerate(tqdm(targets, desc=f"ecg_nn inference ({model_version})", unit="beat"), start=1):
                _notify('inference', i, n_targets)
                x = torch.from_numpy(
                    preprocess_hubert(beat.context_window)
                ).unsqueeze(0).to(device)
                emb = encoder.encode(x)                          # (1, 12, t, 768)

                dw  = torch.from_numpy(beat.decision_window).unsqueeze(0).to(device)
                win = torch.from_numpy(beat.window.astype('float32')).unsqueeze(0).to(device)

                n_leads = dw.shape[1]                            # 12, excludes VD d
                window_2ch = torch.stack([dw, win[:, :n_leads, :]], dim=2)  # (1, L, 2, W)

                win_start_ms = float(beat.spike_idx - beat.window_pre)

                if model_version == 'v6':
                    # v6 is purely parametric -- t_on/tau_on/t_off/tau_off ARE
                    # the prediction. The mask it also returns is a rendering
                    # of those coefs (_coefs_to_mask), not an independent
                    # estimate, so read coefs_b directly instead of running a
                    # mask threshold-crossing heuristic on a derived mask.
                    coefs_b = head(window_2ch, emb)[2]
                    t_on, tau_on, t_off, tau_off = coefs_b[0].tolist()
                    beat.qrs_start        = win_start_ms + t_on
                    beat.qrs_duration     = max(0.0, t_off - t_on)
                    beat.qrs_start_uncert = tau_on
                    beat.qrs_end_uncert   = tau_off
                else:
                    # FiLM MaskHead (v4) has no parametric coefs -- the mask
                    # over the window IS the prediction, so derive start/end
                    # from where it crosses threshold.
                    _, mask, durations, _, _ = head(window_2ch, emb)
                    mask_np = mask[0, 0].cpu().numpy()
                    rel_start, start_uncert, rel_end, end_uncert = mask_to_interval(mask_np)
                    if rel_start is not None:
                        beat.qrs_start        = win_start_ms + rel_start
                        beat.qrs_start_uncert = start_uncert
                        beat.qrs_end_uncert   = end_uncert
                        beat.qrs_duration     = (rel_end - rel_start if rel_end is not None
                                                  else float(durations[0, 0].item()))
                    else:
                        beat.qrs_duration = float(durations[0, 0].item())

                # Quality check for beats that started out overlap-flagged
                # (recovered window for 'recovery', plain window anyway for
                # 'force') and came out with a short duration -- a common
                # symptom of a bad prediction.
                #
                # Scoped to was_noisy ONLY -- a beat that was never
                # overlap-flagged and still comes out short is a genuine
                # model failure, not a recovery artifact, and touching it
                # here would mask that instead of surfacing it. 'exclude' has
                # nothing to check either way: it never predicts on noisy
                # beats in the first place.
                is_suspect = (beat.was_noisy and beat.qrs_duration is not None
                              and beat.qrs_duration < NOISY_BEAT_MIN_DURATION_MS)
                if is_suspect and mode == 'recovery':
                    # Discard: gate it out of the marking output entirely.
                    beat.noisy = True
                elif is_suspect and mode == 'force':
                    # Keep it visible (that's the point of 'force'), but flag
                    # low confidence instead of trusting tau_on/tau_off (or
                    # the FiLM mask-derived uncertainty) as-is.
                    beat.qrs_start_uncert = FORCE_MODE_LOW_CONFIDENCE_UNCERTAINTY
                    beat.qrs_end_uncert   = FORCE_MODE_LOW_CONFIDENCE_UNCERTAINTY

    # ── collections ──────────────────────────────────────────────────────────

    @property
    def beats(self):
        return list(self._beats)

    @property
    def annotated(self):
        return [b for b in self._beats
                if b.qrs_duration is not None and b.qt_interval is not None]

    @property
    def unannotated(self):
        return [b for b in self._beats
                if b.qrs_duration is None or b.qt_interval is None]

    @property
    def noisy(self):
        return [b for b in self._beats if b.noisy]

    def __len__(self):
        return len(self._beats)

    def __iter__(self):
        return iter(self._beats)

    def __getitem__(self, idx):
        return self._beats[idx]

    def __repr__(self):
        return (f"Recording({len(self._beats)} beats, "
                f"{len(self.annotated)} annotated, "
                f"{len(self.noisy)} noisy)")

    # ── ECG_MARKER export ────────────────────────────────────────────────────

    def to_ecg_marker_freq(self):
        """Return period (RR) rows from detected beat positions.

        Works without model predictions — uses spike_idx and BCL only.
        Returns list of [beat_start_ms, beat_end_ms, rr_period_ms]
        matching janela.freq structure.
        """
        rows = []
        for b in self._beats:
            if b.bcl is None:
                continue
            rows.append([float(b.spike_idx - b.bcl), float(b.spike_idx), float(b.bcl)])
        return rows

    def to_ecg_marker_qrs(self):
        """Return QRS annotations in ECG_MARKER format.

        If model predictions are available (qrs_duration set), returns full rows.
        Falls back to spike position with placeholder duration if no model yet.
        Returns list of [qrs_start_ms, qrs_end_ms, rr_period_ms, qrs_duration_ms]
        """
        rows = []
        for b in self._beats:
            period = b.period if b.period is not None else (b.bcl or 0.0)
            if b.qrs_duration is not None and b.qrs_start is not None:
                start = b.qrs_start
                end   = start + b.qrs_duration
                rows.append([start, end, period, b.qrs_duration])
            else:
                start = float(b.spike_idx - b.window_pre)
                end   = float(b.spike_idx + 80)
                rows.append([start, end, period, 80.0])
        return rows

    def to_ecg_marker_qt(self):
        """Return QT annotations in ECG_MARKER format.

        Only populated when model predictions are available.
        Returns list of [qt_start_ms, qt_end_ms, rr_period_ms, qt_duration_ms]
        """
        rows = []
        for b in self._beats:
            if b.qt_interval is None or b.qt_start is None:
                continue
            start  = b.qt_start
            end    = start + b.qt_interval
            period = b.period if b.period is not None else (b.bcl or 0.0)
            rows.append([start, end, period, b.qt_interval])
        return rows


def _synthetic_leads(n=15000, bcl=800, seed=0):
    """12-lead + VD-d synthetic pacing signal for the self-test below.

    Sharp VD-d pacing spikes drive detection (_detect_spikes_from_stimulus)
    so the test doesn't depend on Pan-Tompkins tuning; each spike also gets a
    Gaussian bump on the 12 standard leads so the beat windows aren't just
    noise. Spikes start at 3000 and stop 2500 samples before the end so
    every beat gets a full HuBERT context window (CONTEXT_PRE=CONTEXT_POST=2500).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    spike_positions = list(range(3000, n - 2500, bcl))

    lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    leads = {}
    for li, name in enumerate(lead_names):
        sig = rng.normal(0, 5, n).astype(np.float32)
        for p in spike_positions:
            lo, hi = max(0, p - 20), min(n, p + 20)
            bump = 200 * np.exp(-0.5 * ((np.arange(lo, hi) - p) / 8.0) ** 2)
            sig[lo:hi] += (bump * (1.0 + 0.1 * li)).astype(np.float32)
        leads[name] = sig

    vd = np.zeros(n, dtype=np.float32)
    for p in spike_positions:
        vd[max(0, p - 2):p + 3] = 500.0
    leads['VD d'] = vd

    return leads, spike_positions


def main():
    """Self-test: synthesize a pacing signal, run from_signal(..., predict=True)
    end to end (spike detection -> windows -> HuBERT encoding -> mask-head
    inference), and sanity-check the result. Model version (v6 vs FiLM) is
    whatever Recording._run_inference auto-selects (see its docstring).
    """
    leads, spike_positions = _synthetic_leads()

    rec = Recording.from_signal(leads, predict=True)
    print(rec)
    n_qrs = sum(1 for b in rec.beats if b.qrs_duration is not None)
    print(f"expected spikes: {len(spike_positions)}  detected beats: {len(rec)}  "
          f"noisy: {len(rec.noisy)}  qrs_predicted: {n_qrs}")

    assert len(rec) > 0, "no beats detected"
    # NOTE: rec.annotated requires qt_interval too, which _run_inference does
    # not populate (FiLM or v6) -- QT prediction isn't wired up yet. Check the
    # QRS prediction directly instead.
    assert n_qrs > 0, "no beats got model predictions"
    for b in rec.beats[:3]:
        print(f"  spike={b.spike_idx}  qrs_start={b.qrs_start}  "
              f"qrs_duration={b.qrs_duration}")

    freq_rows = rec.to_ecg_marker_freq()
    qrs_rows  = rec.to_ecg_marker_qrs()
    assert len(freq_rows) > 0 and len(qrs_rows) > 0
    print("OK")


if __name__ == '__main__':
    main()
