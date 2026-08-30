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
    def from_signal(cls, leads, annotations=None, source=None, predict=False):
        """Build a Recording from a leads dict already in memory.

        Parameters
        ----------
        leads       : dict[str, np.ndarray]   lead name → 1-D signal at 1 kHz
        annotations : dict | None             ECG_MARKER annotation dicts;
                                              if None all beats come back unannotated
        source      : str | None              filepath label attached to each Beat
        predict     : bool                    run NN inference to fill qrs/qt fields

        Returns
        -------
        Recording
        """
        annotations = annotations or {}
        matrix, lead_names = leads_matrix(leads)

        if 'VD d' in leads:
            spikes = _detect_spikes_from_stimulus(leads['VD d'])
        else:
            spikes = detect_spikes(leads)
        beats  = extract_windows(matrix, spikes)
        mark_noisy_beats(beats)
        recover_noisy_beats(beats, matrix)
        annotate_beats(beats, annotations)
        compute_bcl(beats)
        extract_context_windows(matrix, beats)
        extract_decision_windows(leads, lead_names, beats)

        if source is not None:
            for b in beats:
                b.source = source

        rec = cls(beats, lead_names, matrix)

        if predict:
            rec._run_inference()

        return rec

    # ── inference ────────────────────────────────────────────────────────────

    def _run_inference(self):
        """Run FiLM MaskHead inference and write qrs_duration / qt_interval
        back into each beat.  Skips noisy beats.
        """
        import torch
        from .model import build_model, build_encoder
        from .dataset import preprocess_hubert
        from .inference import mask_to_interval

        weights_path = os.path.join(os.path.dirname(__file__), 'weights', 'film_head.pt')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"No trained weights at {weights_path}. "
                "Train with train_film.py then re-run export.sh."
            )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        head, _ = build_model(device=device)
        head.load_state_dict(torch.load(weights_path, map_location=device))
        head.eval()

        encoder, _ = build_encoder(device=device, freeze=True)
        encoder.eval()

        targets = [b for b in self._beats if not b.noisy and b.context_window is not None]
        if not targets:
            return

        with torch.no_grad():
            for beat in targets:
                x = torch.from_numpy(
                    preprocess_hubert(beat.context_window)
                ).unsqueeze(0).to(device)
                emb = encoder.encode(x)                          # (1, 12, t, 768)

                dw  = torch.from_numpy(beat.decision_window).unsqueeze(0).to(device)
                win = torch.from_numpy(beat.window.astype('float32')).unsqueeze(0).to(device)

                n_leads = dw.shape[1]                            # 12, excludes VD d
                window_2ch = torch.stack([dw, win[:, :n_leads, :]], dim=2)  # (1, L, 2, W)

                _, mask, durations, _, _ = head(window_2ch, emb)

                # derive start/end ± uncertainty from soft mask
                mask_np = mask[0, 0].cpu().numpy()
                rel_start, start_uncert, rel_end, end_uncert = mask_to_interval(mask_np)

                win_start_ms = float(beat.spike_idx - beat.window_pre)
                if rel_start is not None:
                    beat.qrs_start        = win_start_ms + rel_start
                    beat.qrs_start_uncert = start_uncert
                    beat.qrs_end_uncert   = end_uncert
                    if rel_end is not None:
                        beat.qrs_duration = rel_end - rel_start
                    else:
                        beat.qrs_duration = float(durations[0, 0].item())
                else:
                    beat.qrs_duration = float(durations[0, 0].item())

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
