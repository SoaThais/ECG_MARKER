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

# Production QRS ensemble bundles (weights/, ported from daint's
# logits/production/ -- see that dir's README.md). All three use the same
# architecture as our own v6 port (linear/no-HuBERT tau head); only the
# member list and provenance differ:
#   '4fold'    -- megamodel_loo32.pt, 32 members (4 leave-one-out folds x 8
#                 seeds), from a completed run with real held-out validation.
#   'light'    -- the SAME megamodel_loo32.pt file, subset at load time
#                 (_LIGHT_ENSEMBLE_SEED, see _run_inference) down to 4
#                 members, one per fold (seed 0), instead of 8 -- 8x less
#                 compute per beat, still one vote from every held-out
#                 patient. No separate file to ship or download.
#   'complete' -- allseed64.pt, 64 members trained on all patients, no
#                 holdout -- no honest validation is possible for this one.
# Picked at runtime via Recording.from_signal(..., ensemble_bundle=...).
# There is no fallback model: if the bundle is missing, _run_inference raises.
# The old FiLM (v4) and mid-training v6 heads used to fill that role and were
# removed -- they predicted from different, unvalidated weights, so silently
# degrading to them produced numbers no golden fixture covers.
ENSEMBLE_BUNDLES = ('4fold', 'light', 'complete')
_ENSEMBLE_BUNDLE_FILES = {
    '4fold': 'megamodel_loo32.pt',
    'light': 'megamodel_loo32.pt',   # same file as '4fold' -- subset after load
    'complete': 'allseed64.pt',
}
_LIGHT_ENSEMBLE_SEED = 0  # which seed represents each fold in 'light' mode
_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')


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
                     progress_callback=None, noisy_beat_mode='recovery',
                     ensemble_bundle='4fold'):
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
        ensemble_bundle : str                 one of ENSEMBLE_BUNDLES -- which
                                              production QRS ensemble to use,
                                              if the bundle files are present
                                              (see ENSEMBLE_BUNDLES docstring
                                              above). Ignored otherwise.

        Returns
        -------
        Recording
        """
        if noisy_beat_mode not in NOISY_BEAT_MODES:
            raise ValueError(f"noisy_beat_mode must be one of {NOISY_BEAT_MODES}, "
                              f"got {noisy_beat_mode!r}")
        if ensemble_bundle not in ENSEMBLE_BUNDLES:
            raise ValueError(f"ensemble_bundle must be one of {ENSEMBLE_BUNDLES}, "
                              f"got {ensemble_bundle!r}")

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
        rec.ensemble_bundle = ensemble_bundle

        if predict:
            rec._run_inference(progress_callback=progress_callback)

        return rec

    # ── inference ────────────────────────────────────────────────────────────

    # Model + encoder are expensive to build (HuBERT load, checkpoint read,
    # CUDA init) and don't depend on the recording, so cache them at class
    # level across calls/instances instead of rebuilding on every
    # from_signal(..., predict=True). Cleared automatically if a different
    # process starts (module-level state, not persisted). Keyed by
    # (device, ensemble_bundle) so switching bundles rebuilds correctly.
    _model_cache = {}

    # Beats per inference call. 32 is the sweet spot and re-measured on an
    # RTX 4070 Ti SUPER (megamodel_loo32, 32 members, deferred path):
    # 4.13 ms/beat at B=16, 3.79 at B=32, then WORSE further out -- 4.51 at
    # B=64 and 4.95 at B=128, because the per-layer activations scale with B
    # and the path is memory-bound, so past 32 the extra footprint costs more
    # than the diluted per-call overhead saves. (The older note here claimed a
    # monotonic improvement out to B=102 on the GTX 1080; that was measured
    # before the sync and fusion fixes, when fixed CPU dispatch overhead still
    # dominated.) Also comfortable on an 8GB card. Tune down if predict()
    # ever OOMs.
    _INFERENCE_BATCH_SIZE = 32

    @staticmethod
    def _load_production():
        """Import the vendored loader and the optimized inference path.
        Lazy so that `import ecg_nn` stays cheap and a torch-less
        environment can still load this module for the non-inference
        helpers.
        """
        from .qrs_ensemble import QRSEnsemble
        from .qrs_ensemble_optimized import OptimizedQRSEnsemble
        return QRSEnsemble, OptimizedQRSEnsemble

    @staticmethod
    def _subset_ensemble_to_light(ensemble):
        """'light' mode: keep exactly one member (seed == _LIGHT_ENSEMBLE_SEED)
        per unique held_out fold, in place, using the loaded QRSEnsemble's own
        provenance list -- no edits to the vendored qrs_ensemble.py needed,
        since .predict() only ever iterates self.heads.
        """
        keep = [i for i, p in enumerate(ensemble.provenance)
                if p.get('seed') == _LIGHT_ENSEMBLE_SEED]
        if not keep:
            # Provenance didn't have the expected shape (e.g. a bundle built
            # differently than megamodel_loo32.pt) -- fail safe to the full
            # ensemble rather than silently predicting on zero members.
            return ensemble
        ensemble.heads = [ensemble.heads[i] for i in keep]
        ensemble.provenance = [ensemble.provenance[i] for i in keep]
        return ensemble

    def _run_inference(self, progress_callback=None):
        """Run mask-head inference and write qrs_duration / qt_interval back
        into each beat.  Skips noisy beats.

Uses the production ensemble in weights/ (self.ensemble_bundle) -- a
        median across many MaskHeadV6 members from a completed run, run
        through the optimized path (see qrs_ensemble_optimized.py and
        README.md). It is the only model; a missing bundle raises rather
        than falling back.

        progress_callback, if given, is called as progress_callback(stage, i, n)
        where stage is 'loading_model', 'loading_encoder', or 'inference'
        (i/n are 0/0 for the loading stages; for inference, i is the
        cumulative beat count processed so far -- inference runs in batches
        of _INFERENCE_BATCH_SIZE, so i jumps by up to that many at once,
        not one at a time).
        """
        import torch
        from .encoder import build_encoder
        from .dataset import preprocess_hubert

        def _notify(stage, i=0, n=0):
            if progress_callback is not None:
                progress_callback(stage, i, n)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ensemble_bundle = getattr(self, 'ensemble_bundle', '4fold')
        ensemble_path = os.path.join(_WEIGHTS_DIR, _ENSEMBLE_BUNDLE_FILES[ensemble_bundle])
        use_ensemble = os.path.exists(ensemble_path)
        cache_key = (str(device), ensemble_bundle if use_ensemble else None)

        if cache_key in self._model_cache:
            head, model_version, encoder = self._model_cache[cache_key]
        elif use_ensemble:
            _notify('loading_model')
            QRSEnsemble, OptimizedQRSEnsemble = self._load_production()
            head = QRSEnsemble.load(ensemble_path, device=str(device))
            if ensemble_bundle == 'light':
                head = self._subset_ensemble_to_light(head)
            # Wrap in the optimized path: deferred sync + torch.compile +
            # fp16 on every submodule that passes the golden fixture (the
            # backbones stay fp32 -- see qrs_ensemble_optimized.py). 2.09x
            # over the plain loop on an RTX 4070 Ti SUPER, worst-case golden
            # deviation 0.13 ms against a 0.5 ms tolerance. Degrades to the
            # deferred path on CPU or if compile fails, never to wrong
            # numbers. The first call pays ~1-2 min of compilation, which is
            # why the result is held in _model_cache for the session.
            head = OptimizedQRSEnsemble.from_ensemble(head)
            model_version = 'ensemble'

            _notify('loading_encoder')
            encoder, _ = build_encoder(device=device, freeze=True)
            encoder.eval()

            self._model_cache[cache_key] = (head, model_version, encoder)
        else:
            raise FileNotFoundError(
                f"No QRS ensemble bundle at {ensemble_path}. ecg_nn ships exactly "
                f"one model -- the production ensemble in weights/ (see README.md); "
                f"the old FiLM and mid-training v6 fallbacks were removed because "
                f"they predicted from different, unvalidated weights."
            )

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

        n_targets = len(targets)
        _notify('inference', 0, n_targets)

        # Fixed-shape batches (same window size, same batch size for every
        # call but the last) run many times over -- once per progress step,
        # 32/64x more per call again inside the ensemble's own member loop --
        # so cudnn's autotuner (picks the fastest conv algorithm after a short
        # warmup, keyed on input shape) actually pays for itself here, unlike
        # a model that sees a different shape every call.
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        processed = 0
        with torch.no_grad():
            for batch_start in range(0, n_targets, self._INFERENCE_BATCH_SIZE):
                batch = targets[batch_start:batch_start + self._INFERENCE_BATCH_SIZE]

                # HuBERT encoding batched too -- it's a transformer forward
                # pass per lead, i.e. the expensive half of each beat, not
                # just the mask head. encoder.encode() already reshapes
                # (N, L, T) -> (N*L, T) internally, so this was always
                # batch-ready; only the caller wasn't using it.
                x = torch.from_numpy(
                    np.stack([preprocess_hubert(b.context_window) for b in batch])
                ).to(device)
                emb = encoder.encode(x)                          # (B, 12, t, 768)

                dw  = torch.from_numpy(np.stack([b.decision_window for b in batch])).to(device)
                win = torch.from_numpy(
                    np.stack([b.window.astype('float32') for b in batch])
                ).to(device)

                n_leads = dw.shape[1]                            # 12, excludes VD d
                window_2ch = torch.stack([dw, win[:, :n_leads, :]], dim=2)  # (B, L, 2, W)

                win_start_ms = np.array(
                    [float(b.spike_idx - b.window_pre) for b in batch], dtype=np.float64)

                # QRSEnsemble.predict() does its own median-across-members
                # combination (median(onset), median(offset), duration =
                # their difference -- NOT the median of per-member durations,
                # see README.md) and returns tau_on/tau_off as genuine model
                # outputs. Already batch-native.
                out = head.predict(window_2ch, emb)
                onset, duration = out['onset'], out['duration']
                tau_on, tau_off = out['tau_on'], out['tau_off']
                for j, beat in enumerate(batch):
                    beat.qrs_start        = win_start_ms[j] + float(onset[j])
                    beat.qrs_duration     = max(0.0, float(duration[j]))
                    beat.qrs_start_uncert = float(tau_on[j])
                    beat.qrs_end_uncert   = float(tau_off[j])

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
                for beat in batch:
                    is_suspect = (beat.was_noisy and beat.qrs_duration is not None
                                  and beat.qrs_duration < NOISY_BEAT_MIN_DURATION_MS)
                    if is_suspect and mode == 'recovery':
                        # Discard: gate it out of the marking output entirely.
                        beat.noisy = True
                    elif is_suspect and mode == 'force':
                        # Keep it visible (that's the point of 'force'), but
                        # flag low confidence instead of trusting tau_on/
                        # tau_off as-is.
                        beat.qrs_start_uncert = FORCE_MODE_LOW_CONFIDENCE_UNCERTAINTY
                        beat.qrs_end_uncert   = FORCE_MODE_LOW_CONFIDENCE_UNCERTAINTY

                processed += len(batch)
                _notify('inference', processed, n_targets)

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
    end to end (spike detection -> windows -> HuBERT encoding -> ensemble
    inference), and sanity-check the result. Requires a bundle in weights/.
    """
    leads, spike_positions = _synthetic_leads()

    rec = Recording.from_signal(leads, predict=True)
    print(rec)
    n_qrs = sum(1 for b in rec.beats if b.qrs_duration is not None)
    print(f"expected spikes: {len(spike_positions)}  detected beats: {len(rec)}  "
          f"noisy: {len(rec.noisy)}  qrs_predicted: {n_qrs}")

    assert len(rec) > 0, "no beats detected"
    # NOTE: rec.annotated requires qt_interval too, which _run_inference does
    # not populate -- QT prediction isn't wired up yet. Check the
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
