"""ECG model — FiLM MaskHead + HuBERT-ECG encoder + PTHead baseline + MaskHeadV1 (legacy).

FiLM MaskHead (primary model)
------------------------------
Inputs : window  (N, L, 2, W)       — per-lead 2-ch beat window: [PT decision, raw ECG]
         context (N, L, t, embed_dim) — per-lead HuBERT embeddings (t=38, embed_dim=768)
Output : (N, 1, W)                  — soft probability mask over the beat window
                                      sum over last dim ≈ duration in ms

MaskHeadV1 (legacy)
---------------------
Inputs : x        (N, L, t, D)  — HuBERT embeddings
         decision (N, n_pt_leads, W) — Pan-Tompkins integrated signal
Output : (N, 1, W) logits, (N, 1, W) mask, (N, 1) durations
Original compressor + dilated-fusion architecture (no FiLM conditioning).

HuBERTECGRegressor (encoder)
------------------------------
Input  : (N, 12, 2500)  — 12-lead, 500 Hz, 5-second context window
Output : (N, 12, t, D)  — per-lead HuBERT embeddings via .encode()

PTHead (baseline)
------------------
Ignores context entirely; predicts from the PT channel of the beat window only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODER_DIM = 32


class _Positive(nn.Module):
    def forward(self, x):
        return x.abs()


# =========================================================
# FiLM MaskHead  (primary model)
# =========================================================

class MaskHead(nn.Module):
    FILM_CHANNELS = 64

    def __init__(self, window_size=None, n_leads=12, embed_dim=768, scale=1, **kwargs):
        super().__init__()
        if window_size is None:
            from .beat import WINDOW_PRE, WINDOW_POST
            window_size = WINDOW_PRE + WINDOW_POST
        self.window_size = window_size
        self.n_leads = n_leads
        self.scale = scale
        C = self.FILM_CHANNELS * scale

        # input: (N, embed_dim, n_leads, t)  output: (N, ENCODER_DIM, 1, t)
        self.compress = nn.Sequential(
            nn.Conv2d(embed_dim, 4*C, kernel_size=1),
            nn.BatchNorm2d(C*4),
            nn.GELU(),
            nn.Conv2d(4*C, 2*C, kernel_size=1),
            nn.Conv2d(2*C, 2*C, kernel_size=(n_leads, 1)),
        )
        self.film_conv = nn.Sequential(
            nn.Conv1d(2*C, C*2, kernel_size=3, padding=1),
        )

        self.fusion_pre = nn.Sequential(
            nn.Conv1d(n_leads * 2, C, kernel_size=7, padding=3), nn.GELU(),
        )
        self.fusion_post = nn.Sequential(
            nn.Conv1d(C,   4*C, kernel_size=7, padding=6,  dilation=2), nn.GELU(),
            nn.Conv1d(4*C, C,   kernel_size=7, padding=12, dilation=4), nn.GELU(),
            nn.Conv1d(C,   C,   kernel_size=7, padding=24, dilation=8), nn.GELU(),
            nn.Conv1d(C,   1,   kernel_size=1),
        )

        self.ablate_context  = False
        self.ablate_decision = False
        self.noise_gamma = nn.Parameter(torch.zeros(1, C, window_size))
        self.noise_beta  = nn.Parameter(torch.zeros(1, C, window_size))

    def _norm(self, x):
        mu  = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (x - mu) / std

    def _forward_impl(self, window, context):
        N, L = window.shape[:2]
        W = window.shape[-1]

        g = context.permute(0, 3, 1, 2)
        g = self.compress(g)
        g = g.squeeze(2)
        g = F.interpolate(g, size=self.window_size, mode='linear', align_corners=False)

        gb = self.film_conv(g)
        gamma, beta = gb.chunk(2, dim=1)
        g   = 1 + 0.1 * gamma
        b   = 0.1 * beta

        g_f = 1 + 0.1 * self.noise_gamma
        b_f = 0.1 * self.noise_beta

        f = window.reshape(N, L * 2, -1)
        f = self._norm(f)
        if self.ablate_decision:
            f = torch.zeros_like(f)

        h = self.fusion_pre(f)
        h = self._norm(h)

        logits_f = self.fusion_post(g_f * h + b_f)
        logits   = self.fusion_post(g * h + b)
        mask     = torch.sigmoid(logits)

        with torch.no_grad():
            pred_diff = (logits - logits_f).detach().abs().mean().item()
        film_m = {'pred_diff': pred_diff}

        return logits, mask, mask.sum(dim=-1), f, logits_f, film_m

    def forward(self, window, context):
        logits, mask, durations, _, logits_f, film_m = self._forward_impl(window, context)
        return logits, mask, durations, logits_f, film_m

    def forward_debug(self, window, context):
        return self._forward_impl(window, context)

    def set_ablation(self, context=False, decision=False):
        self.ablate_context  = context
        self.ablate_decision = decision


# =========================================================
# MaskHeadV1  (legacy — compressor + dilated fusion, no FiLM)
# =========================================================

class MaskHeadV1(nn.Module):
    """Original mask head: HuBERT compressor + PT lead-mix + dilated fusion stack.

    Inputs : x        (N, L, t, D)       — HuBERT embeddings
             decision (N, n_pt_leads, W) — PT integrated signal per lead
    Output : logits (N, 1, W), mask (N, 1, W), durations (N, 1)
    """

    def __init__(self, embed_dim=768, window_size=None, width=128, n_pt_leads=12):
        super().__init__()
        if window_size is None:
            from .beat import WINDOW_PRE, WINDOW_POST
            window_size = WINDOW_PRE + WINDOW_POST
        self.window_size = window_size

        self.compress = nn.Sequential(
            nn.Conv2d(embed_dim, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.GELU(),
            nn.Conv2d(1024, 512, kernel_size=1), nn.GELU(),
            nn.Conv2d(512, 128, kernel_size=(1, 7), padding=(0, 3), groups=128), nn.GELU(),
            nn.Conv2d(128, 64,  kernel_size=(1, 7), padding=(0, 3), groups=64),  nn.GELU(),
            nn.Conv2d(64,  32,  kernel_size=(1, 7), padding=(0, 3), groups=32),  nn.GELU(),
            nn.Conv2d(32,  16,  kernel_size=(1, 7), padding=(0, 3), groups=16),  nn.GELU(),
            nn.Conv2d(16,   1,  kernel_size=(12, 7), padding=(0, 3)),
        )

        self.pt_lead_w = nn.Parameter(torch.ones(n_pt_leads))
        self.pt_lead_b = nn.Parameter(torch.zeros(n_pt_leads))
        nn.utils.parametrize.register_parametrization(self, 'pt_lead_w', _Positive())
        nn.utils.parametrize.register_parametrization(self, 'pt_lead_b', _Positive())

        self.fusion = nn.Sequential(
            nn.Conv1d(2,   128, kernel_size=7, padding=3,  dilation=1), nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=7, padding=6,  dilation=2), nn.GELU(),
            nn.Conv1d(256, 128, kernel_size=7, padding=12, dilation=4), nn.GELU(),
            nn.Conv1d(128, 64,  kernel_size=7, padding=24, dilation=8), nn.GELU(),
            nn.Conv1d(64,  1,   kernel_size=1),
        )

    def _forward_impl(self, x, decision):
        N, L, t, D = x.shape
        g = x.permute(0, 3, 1, 2)                              # (N, D, L, t)
        g = self.compress(g)                                    # (N, 1, 1, t)
        g = g.squeeze(2)                                        # (N, 1, t)
        g = F.interpolate(g, size=self.window_size, mode='linear', align_corners=False)
        mu  = g.mean(dim=-1, keepdim=True)
        std = g.std(dim=-1, keepdim=True).clamp(min=1e-6)
        g = (g - mu) / std

        w = self.pt_lead_w.view(1, -1, 1)
        b = self.pt_lead_b.view(1, -1, 1)
        f = (w * decision + b).sum(dim=1, keepdim=True)
        mu  = f.mean(dim=-1, keepdim=True)
        std = f.std(dim=-1, keepdim=True).clamp(min=1e-6)
        f = (f - mu) / std

        logits = self.fusion(torch.cat([g, f], dim=1))
        mask   = torch.sigmoid(logits)
        return logits, mask, mask.sum(dim=-1)

    def forward(self, x, decision):
        return self._forward_impl(x, decision)

    def forward_debug(self, x, decision):
        return self._forward_impl(x, decision)


# =========================================================
# HuBERT-ECG encoder
# =========================================================

class HuBERTECGRegressor(nn.Module):
    """Frozen HuBERT-ECG encoder.  Call .encode(x) to get per-lead embeddings."""

    def __init__(self, repo_id='Edoardo-BS/hubert-ecg-base', freeze=True, **kwargs):
        super().__init__()
        from transformers import AutoModel
        from .beat import WINDOW_PRE, WINDOW_POST

        self.encoder = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
        embed_dim = self.encoder.config.hidden_size

        with torch.no_grad():
            dummy = torch.zeros(1, 12, 2500)
            N, L, T = dummy.shape
            out = self.encoder(input_values=dummy.reshape(N * L, T)).last_hidden_state
            self.t = out.shape[1]
            self.L = L

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def encode(self, x):
        """x: (N, 12, 2500) → (N, 12, t, D)"""
        N, L, T = x.shape
        out = self.encoder(input_values=x.reshape(N * L, T)).last_hidden_state
        t, D = out.shape[1], out.shape[2]
        return out.reshape(N, L, t, D)


# =========================================================
# PT baseline head
# =========================================================

class PTHead(nn.Module):
    """Baseline: ignores context, uses only the PT channel (ch 0) of window."""

    def __init__(self, window_size=None, **kwargs):
        super().__init__()
        if window_size is None:
            from .beat import WINDOW_PRE, WINDOW_POST
            window_size = WINDOW_PRE + WINDOW_POST
        self.window_size = window_size
        self.scale = nn.Linear(window_size, window_size)

    def _forward_impl(self, window, context):
        f = window[:, :, 0, :].mean(dim=1, keepdim=True)
        mu  = f.mean(dim=-1, keepdim=True)
        std = f.std(dim=-1, keepdim=True).clamp(min=1e-6)
        f_norm = (f - mu) / std
        logits = self.scale(f_norm)
        mask   = torch.sigmoid(logits)
        return logits, mask, mask.sum(dim=-1), f_norm, None

    def forward(self, window, context):
        logits, mask, durations, _, _ = self._forward_impl(window, context)
        return logits, mask, durations, None

    def forward_debug(self, window, context):
        return self._forward_impl(window, context)


# =========================================================
# Builders
# =========================================================

def build_model(device=None, **kwargs):
    """Build the FiLM MaskHead (primary model)."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MaskHead(**kwargs).to(device)
    return model, device


def build_encoder(device=None, **kwargs):
    """Build the HuBERT-ECG encoder."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HuBERTECGRegressor(**kwargs).to(device)
    return model, device
