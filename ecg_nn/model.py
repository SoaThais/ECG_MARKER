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

import math

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


# =========================================================
# MaskHeadV6  (attention-pooling architecture, ported from
# logits/src/v6/model_stage2.py on daint — see ../v6_daint/)
# =========================================================

class LeadFeatureCompressor(nn.Module):
    """Summarizes pre-pooling per-lead backbone features into a fixed embedding.

    Input h_leads: (N, L, C, W) — the SAME per-lead conv output the scorer reads,
    before attention pooling collapses it. Mean+max pooled over time per lead,
    flattened across leads, then projected.
    """
    def __init__(self, C, L=12, out_dim=256, hidden=128):
        super().__init__()
        self.norm = nn.BatchNorm1d(L * 2 * C)
        self.proj = nn.Sequential(
            nn.Linear(L * 2 * C, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h_leads):                # (N, L, C, W)
        mean_p = h_leads.mean(dim=-1)           # (N, L, C)
        max_p  = h_leads.amax(dim=-1)           # (N, L, C)
        pooled = torch.cat([mean_p, max_p], dim=-1).flatten(1)   # (N, L*2C)
        return self.proj(self.norm(pooled))     # (N, out_dim)


class RoutingCompressor(nn.Module):
    """Summarizes HuBERT-derived per-lead routing weights into a fixed embedding.

    Input w: (N, L) — softmax'd routing weights (w_on or w_off from the compress
    path). Entropy and top-1 concentration are fed explicitly alongside the raw
    vector.
    """
    def __init__(self, L=12, out_dim=64, hidden=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(L + 2, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, w):                       # (N, L)
        entropy = -(w.clamp(min=1e-10) * w.clamp(min=1e-10).log()).sum(-1, keepdim=True)
        top1    = w.amax(dim=-1, keepdim=True)
        return self.proj(torch.cat([w, entropy, top1], dim=-1))


class MaskHeadV6(nn.Module):
    """Attention-pooling successor to MaskHead (FiLM). Same (window, context)
    input contract; forward() returns an 11-tuple (see forward() below) instead
    of MaskHead's 5-tuple — the two are NOT drop-in interchangeable at the
    caller. Ported verbatim from logits/src/v6/model_stage2.py.

    Two gradient paths:
      PATH A: per-lead backbone → scorer → per-lead t + shared tau — auxiliary NLL
      PATH B: pooled attention → final t + shared tau — main NLL

    tau_detach (default True upstream): whether tau's inputs are detached from
    the backbone/compress path. The ported checkpoint (pre_s2_ep1000.pt) was
    trained with tau_no_detach, i.e. tau_detach=False — see build_model_v6().
    """
    FILM_CHANNELS = 32
    _C = 2.0 * math.log(19)

    def __init__(self, window_size=None, embed_dim=768, scale=1, scorer_scale=1,
                 tau_cap=None, tau_floor=1.0, tau_c_init_on=1.0, tau_c_init_off=10.0,
                 tau_detach=True, offset_prior=True, no_compress=False,
                 compress_mode='full', detach_routing=False, center_context=False, use_ste=False,
                 feature_pool=False, tau_arch='mlp64', tau_feats='all', **kwargs):
        super().__init__()
        if window_size is None:
            from .beat import WINDOW_PRE, WINDOW_POST
            window_size = WINDOW_PRE + WINDOW_POST
        self.window_size  = window_size
        self.tau_cap      = tau_cap
        self.tau_floor    = tau_floor
        self.tau_detach   = tau_detach
        self.offset_prior = offset_prior
        self.scale        = scale
        self.scorer_scale = scorer_scale
        self.no_compress      = no_compress
        self.detach_routing   = detach_routing
        self.center_context   = center_context
        self.use_ste          = use_ste
        self.feature_pool     = feature_pool
        self.tau_arch         = tau_arch
        self.tau_feats        = tau_feats
        if feature_pool and detach_routing:
            raise ValueError('feature_pool is incompatible with detach_routing: it would cut '
                             'the backbone off from the pooled loss, removing the contamination '
                             'path that feature pooling exists to create.')
        C  = int(self.FILM_CHANNELS * scale)
        SC = max(1, round(scorer_scale * C))

        if not no_compress:
            if compress_mode == 'full':
                self.compress = nn.Sequential(
                    nn.Conv2d(embed_dim, C, kernel_size=(1, 1)),
                    nn.BatchNorm2d(C),
                    nn.GELU(),
                    nn.Conv2d(C, C, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
                    nn.GELU(),
                    nn.Conv2d(C, C, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
                    nn.GELU(),
                    nn.Conv2d(C, C, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
                    nn.GELU(),
                    nn.Conv2d(C, 2, kernel_size=(1, 5)),  # 2 channels: onset / offset
                )
            elif compress_mode == 'small':
                self.compress = nn.Sequential(
                    nn.Conv2d(embed_dim, 4, kernel_size=(1, 1)),
                    nn.GELU(),
                    nn.Conv2d(4, 4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d((None, 1)),
                    nn.Conv2d(4, 2, kernel_size=(1, 1)),
                )
            elif compress_mode == 'avgpool':
                self.compress = nn.Sequential(
                    nn.AdaptiveAvgPool2d((None, 1)),
                    nn.Conv2d(embed_dim, 2, kernel_size=(1, 1)),
                )
            else:
                raise ValueError(f'Unknown compress_mode: {compress_mode!r}')

        def _make_backbone():
            return nn.Sequential(
                nn.Conv1d(2,    C//4, 3, groups=2, padding=1,  dilation=1,  padding_mode='reflect'), nn.GELU(),
                nn.Conv1d(C//4, C//4, 3, groups=2, padding=2,  dilation=2,  padding_mode='reflect'), nn.GELU(),
                nn.Conv1d(C//4, C//2, 3, groups=2, padding=4,  dilation=4,  padding_mode='reflect'), nn.GELU(),
                nn.Conv1d(C//2, C//2, 3, groups=2, padding=8,  dilation=8,  padding_mode='reflect'), nn.GELU(),
                nn.Conv1d(C//2, C,    3, groups=2, padding=16, dilation=16, padding_mode='reflect'), nn.GELU(),
                nn.Conv1d(C,    C,    3, groups=2, padding=32, dilation=32, padding_mode='reflect'), nn.GELU(),
            )

        self.onset_backbone  = _make_backbone()
        self.offset_backbone = _make_backbone()

        def _make_scorer():
            return nn.Sequential(
                nn.Conv1d(C,    SC*2, 7, padding=3, padding_mode='reflect'), nn.GELU(),
                nn.Conv1d(SC*2, SC,   3, padding=1, padding_mode='reflect'), nn.GELU(),
                nn.Conv1d(SC,   C//4, 3, padding=1, padding_mode='reflect'), nn.GELU(),
                nn.Conv1d(C//4, 1,    3, padding=1, padding_mode='reflect'),
            )

        self.onset_scorer  = _make_scorer()
        self.offset_scorer = _make_scorer()

        if tau_feats == 'all':
            TAU_IN = 5 + 256 + 64
        elif tau_feats == 'no_hubert':
            TAU_IN = 5 + 256
        elif tau_feats == 'hand':
            TAU_IN = 5
        else:
            raise ValueError('unknown tau_feats: %r' % tau_feats)

        def _make_tau_head():
            if tau_arch == 'mlp64':
                return nn.Sequential(
                    nn.BatchNorm1d(TAU_IN),
                    nn.Linear(TAU_IN, 64), nn.GELU(),
                    nn.Linear(64, 1),
                )
            if tau_arch == 'mlp16':
                return nn.Sequential(
                    nn.BatchNorm1d(TAU_IN),
                    nn.Linear(TAU_IN, 16), nn.GELU(),
                    nn.Linear(16, 1),
                )
            if tau_arch == 'linear':
                return nn.Sequential(
                    nn.BatchNorm1d(TAU_IN),
                    nn.Linear(TAU_IN, 1),
                )
            raise ValueError('unknown tau_arch: %r' % tau_arch)

        self.onset_tau_head  = _make_tau_head()
        self.offset_tau_head = _make_tau_head()

        self.lead_feat_compressor_on  = LeadFeatureCompressor(C)
        self.lead_feat_compressor_off = LeadFeatureCompressor(C)
        self.routing_compressor_on    = RoutingCompressor()
        self.routing_compressor_off   = RoutingCompressor()

        self.tau_c_on  = nn.Parameter(torch.tensor(float(tau_c_init_on)))
        self.tau_c_off = nn.Parameter(torch.tensor(float(tau_c_init_off)))

        for head in (self.onset_tau_head, self.offset_tau_head):
            _last = max(i for i, m in enumerate(head) if isinstance(m, nn.Linear))
            nn.init.zeros_(head[_last].weight)
            nn.init.zeros_(head[_last].bias)

        t = torch.arange(window_size, dtype=torch.float32).unsqueeze(0)
        self.register_buffer('t_grid', t)

    def _norm(self, x):
        mu  = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (x - mu) / std

    def _coefs_to_mask(self, t_on, tau_on, t_off, tau_off):
        t    = self.t_grid
        rise = torch.sigmoid(self._C * (t - t_on.unsqueeze(-1))   / tau_on.unsqueeze(-1))
        fall = torch.sigmoid(self._C * (t_off.unsqueeze(-1) - t)  / tau_off.unsqueeze(-1))
        return (rise * fall).unsqueeze(1)

    def _make_off_bias(self, t_on, W, device):
        t_on_idx = t_on.long().clamp(0, W - 1)
        pos      = torch.arange(W, device=device).unsqueeze(0)
        return -(t_on_idx.unsqueeze(1) - pos).clamp(min=0).float()

    def _softargmax(self, score):
        attn   = F.softmax(score, dim=-1)
        t_soft = (attn * self.t_grid).sum(-1)
        if self.use_ste:
            idx    = score.argmax(dim=-1)
            t_hard = self.t_grid[0, idx]
            t_soft = t_hard + (t_soft - t_soft.detach())
        return attn, t_soft

    def _maybe_detach(self, x):
        return x.detach() if self.tau_detach else x

    def _tau_from_all(self, tau_head, lead_compressor, routing_compressor,
                       attn_per_lead, t_leads, h_leads, w, tau_c):
        W = self.window_size

        attn_d = self._maybe_detach(attn_per_lead)
        t_d    = self._maybe_detach(t_leads)

        pooled_uni = attn_d.mean(dim=1)  # (N, W)
        H_pooled_norm = (-(pooled_uni * pooled_uni.clamp(min=1e-10).log()).sum(-1)
                         / math.log(W)).unsqueeze(-1)                                 # (N,1)
        H_l       = -(attn_d * attn_d.clamp(min=1e-10).log()).sum(-1)  # (N,L)
        H_mean    = (H_l.mean(dim=-1) / math.log(W)).unsqueeze(-1)                   # (N,1)
        H_std     = (H_l.std(dim=-1)  / math.log(W)).unsqueeze(-1)                   # (N,1)
        var_t     = (t_d.var(dim=-1).clamp(min=0) / (W ** 2)).unsqueeze(-1)          # (N,1)
        peak_leads = attn_d.max(dim=-1).values.mean(dim=-1, keepdim=True)            # (N,1)
        hand_feat = torch.cat([H_pooled_norm, H_mean, H_std, var_t, peak_leads], dim=-1)  # (N,5)

        feat_lead = lead_compressor(self._maybe_detach(h_leads))    # (N,256)
        feat_hub  = routing_compressor(self._maybe_detach(w))       # (N,64)

        if self.tau_feats == 'all':
            _parts = [feat_lead, feat_hub, hand_feat]
        elif self.tau_feats == 'no_hubert':
            _parts = [feat_lead, hand_feat]
        else:
            _parts = [hand_feat]
        feat    = torch.cat(_parts, dim=-1)   # hand_feat is always last
        tau_b   = tau_head(feat)[:, 0]           # per-beat deviation, 0 at t=0
        tau     = (tau_c + tau_b).clamp(min=1.0)  # safety floor only -- tau_c carries the prior
        if self.tau_cap is not None:
            tau = tau.clamp(max=self.tau_cap)
        return tau

    def _forward_impl(self, window, context, routing_alpha=1.0):
        N, L, _, W = window.shape
        C = int(self.FILM_CHANNELS * self.scale)

        w          = window.reshape(N * L, 2, W)
        w          = torch.stack([w[:, 0], w[:, 1].abs()], dim=1)
        f          = self._norm(w)
        h_on_flat  = self._norm(self.onset_backbone(f))    # (N*L, C, W)
        h_off_flat = self._norm(self.offset_backbone(f))

        score_on = self.onset_scorer(h_on_flat).squeeze(1)   # (N*L, W)
        attn_on_flat, t_on_flat = self._softargmax(score_on) # (N*L, W), (N*L,)

        if self.offset_prior:
            off_bias = self._make_off_bias(t_on_flat.detach(), W, f.device)   # (N*L, W)
        else:
            off_bias = None

        score_off = self.offset_scorer(h_off_flat).squeeze(1)   # (N*L, W)
        if off_bias is not None:
            score_off = score_off + off_bias
        attn_off_flat, t_off_flat = self._softargmax(score_off)

        attn_on_per_lead  = attn_on_flat.view(N, L, W)    # (N, L, W)
        attn_off_per_lead = attn_off_flat.view(N, L, W)
        t_on_leads        = t_on_flat.view(N, L)           # (N, L)
        t_off_leads       = t_off_flat.view(N, L)

        if self.no_compress:
            w_on  = torch.full((N, L), 1.0 / L, device=f.device, dtype=f.dtype)
            w_off = w_on
            film_m = {
                'film_gamma_dev': torch.tensor(0.0),
                'film_beta_dev':  torch.tensor(0.0),
                'film_ratio':     torch.tensor(0.0),
            }
        else:
            ctx = context - context.mean(dim=1, keepdim=True) if self.center_context else context
            g         = self.compress(ctx.permute(0, 3, 1, 2)).squeeze(-1)  # (N, 2, L)
            w_on      = F.softmax(g[:, 0, :] * routing_alpha, dim=-1)   # (N, L)
            w_off     = F.softmax(g[:, 1, :] * routing_alpha, dim=-1)

            with torch.no_grad():
                film_m = {
                    'film_gamma_dev': w_on.float().std(dim=-1).mean(),
                    'film_beta_dev':  w_off.float().std(dim=-1).mean(),
                    'film_ratio':     ((w_on.float().var() + w_off.float().var())
                                       / (2.0 / L ** 2 + 1e-8)),
                }

        h_on_leads_flat  = h_on_flat.view(N, L, C, W)
        h_off_leads_flat = h_off_flat.view(N, L, C, W)
        tau_on  = self._tau_from_all(self.onset_tau_head,  self.lead_feat_compressor_on,  self.routing_compressor_on,
                                      attn_on_per_lead,  t_on_leads,  h_on_leads_flat,  w_on,  self.tau_c_on)   # (N,)
        tau_off = self._tau_from_all(self.offset_tau_head, self.lead_feat_compressor_off, self.routing_compressor_off,
                                      attn_off_per_lead, t_off_leads, h_off_leads_flat, w_off, self.tau_c_off)

        coefs_a = torch.stack([
            t_on_leads,  tau_on.unsqueeze(-1).expand(N, L),
            t_off_leads, tau_off.unsqueeze(-1).expand(N, L),
        ], dim=-1)  # (N, L, 4)

        if self.feature_pool:
            h_on_leads  = h_on_flat.view(N, L, C, W)
            h_off_leads = h_off_flat.view(N, L, C, W)
            hf_on  = self._norm((h_on_leads  * w_on.view(N, L, 1, 1)).sum(dim=1))   # (N, C, W)
            hf_off = self._norm((h_off_leads * w_off.view(N, L, 1, 1)).sum(dim=1))

            score_on_b = self.onset_scorer(hf_on).squeeze(1)                        # (N, W)
            pooled_attn_on, t_on = self._softargmax(score_on_b)

            score_off_b = self.offset_scorer(hf_off).squeeze(1)                     # (N, W)
            if self.offset_prior:
                score_off_b = score_off_b + self._make_off_bias(t_on.detach(), W, f.device)
            pooled_attn_off, t_off = self._softargmax(score_off_b)
        else:
            _attn_on  = attn_on_per_lead.detach()  if self.detach_routing else attn_on_per_lead
            _attn_off = attn_off_per_lead.detach() if self.detach_routing else attn_off_per_lead
            pooled_attn_on  = (w_on.unsqueeze(-1)  * _attn_on).sum(dim=1)   # (N, W)
            pooled_attn_off = (w_off.unsqueeze(-1) * _attn_off).sum(dim=1)
            t_on  = (pooled_attn_on  * self.t_grid).sum(-1)   # (N,)
            t_off = (pooled_attn_off * self.t_grid).sum(-1)

        coefs_b = torch.stack([t_on, tau_on, t_off, tau_off], dim=-1)   # (N, 4)

        mask      = self._coefs_to_mask(coefs_b[:, 0], coefs_b[:, 1], coefs_b[:, 2], coefs_b[:, 3])
        durations = (coefs_b[:, 2] - coefs_b[:, 0]).clamp(min=0).unsqueeze(-1)

        h_leads_on  = h_on_flat.view(N, L, C, W).permute(0, 2, 1, 3)
        h_leads_off = h_off_flat.view(N, L, C, W).permute(0, 2, 1, 3)

        return (mask, durations, coefs_b,
                pooled_attn_on, pooled_attn_off, w_on, w_off,
                h_leads_on, h_leads_off,
                film_m, coefs_a,
                attn_on_per_lead, attn_off_per_lead)

    def forward(self, window, context, routing_alpha=1.0, **kwargs):
        (mask, dur, coefs_b, on_attn, off_attn, w_on, w_off,
         _, _, film_m, coefs_a,
         attn_on_pl, attn_off_pl) = self._forward_impl(window, context, routing_alpha=routing_alpha)
        return mask, dur, coefs_b, on_attn, off_attn, w_on, w_off, film_m, coefs_a, attn_on_pl, attn_off_pl

    def forward_debug(self, window, context, routing_alpha=1.0, **kwargs):
        return self._forward_impl(window, context, routing_alpha=routing_alpha)


# Exact construction kwargs for runs/v6_lin_nohub_s2at3000 (recipe.txt on daint),
# the run pre_s2_ep1000.pt was ported from.
V6_LIN_NOHUB_S2AT3000_KWARGS = dict(
    embed_dim=768, scale=1.0, scorer_scale=0.5, compress_mode='full',
    no_compress=False, offset_prior=False, tau_c_init_on=15.0, tau_c_init_off=15.0,
    tau_detach=False, tau_arch='linear', tau_feats='no_hubert',
    center_context=False, use_ste=False, feature_pool=False, detach_routing=False,
)


def build_model_v6(checkpoint_path=None, device=None, **kwargs):
    """Build MaskHeadV6, optionally loading a checkpoint (state_dict, .pt).

    Defaults match runs/v6_lin_nohub_s2at3000/cells/p9/head0_seed0/pre_s2_ep1000.pt
    (see V6_LIN_NOHUB_S2AT3000_KWARGS); pass overrides via kwargs for a
    different run.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    build_kwargs = {**V6_LIN_NOHUB_S2AT3000_KWARGS, **kwargs}
    model = MaskHeadV6(**build_kwargs).to(device)
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    return model, device


def _default_v6_checkpoint():
    import os
    return os.path.join(os.path.dirname(__file__), '..', 'v6_daint', 'model', 'pre_s2_ep1000.pt')


def main():
    """Self-test: build MaskHeadV6, load the ported v6 checkpoint, run a dummy
    forward pass, and sanity-check output shapes/ranges."""
    import os

    ckpt_path = _default_v6_checkpoint()
    if not os.path.exists(ckpt_path):
        print(f"[skip] checkpoint not found: {ckpt_path}")
        return

    model, device = build_model_v6(checkpoint_path=ckpt_path)
    model.eval()
    print(f"MaskHeadV6 loaded on {device}, {sum(p.numel() for p in model.parameters())} params")

    N, L, W, T, D = 2, 12, model.window_size, 38, 768
    window  = torch.randn(N, L, 2, W, device=device)
    context = torch.randn(N, L, T, D, device=device)

    with torch.no_grad():
        (mask, dur, coefs_b, on_attn, off_attn, w_on, w_off,
         film_m, coefs_a, attn_on_pl, attn_off_pl) = model(window, context)

    assert mask.shape == (N, 1, W), mask.shape
    assert dur.shape == (N, 1), dur.shape
    assert coefs_b.shape == (N, 4), coefs_b.shape
    assert torch.isfinite(mask).all()
    assert torch.isfinite(dur).all()

    print(f"mask {tuple(mask.shape)}  durations(ms) {dur.squeeze(-1).tolist()}")
    print(f"t_on/tau_on/t_off/tau_off (beat 0): {coefs_b[0].tolist()}")
    print("OK")


if __name__ == '__main__':
    main()
