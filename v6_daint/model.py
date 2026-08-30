"""MaskHeadV6 — attention-pooling architecture.

Key design:
  - Backbone runs per lead: (N*L, 2, W) → (N*L, C, W)
  - Scorer runs per lead → per-lead attention maps (N, L, W)
  - Attention maps are POOLED across leads (uniform or compress-weighted)
  - Soft-argmax on pooled attention → final t_on, t_off

Two gradient paths:
  PATH A: per-lead backbone → scorer → per-lead t + shared tau — auxiliary NLL
  PATH B: pooled attention → final t + shared tau — main NLL
          no_compress: uniform pool; compress: HuBERT-weighted pool

Tau head (beat-level, shared across PATH A and PATH B):
  Input: [H_pooled_norm, H_mean_l, H_std_l, var_t_l, peak_pooled] (5 hand-crafted
  entropy scalars) + LeadFeatureCompressor(h_leads) (256) + RoutingCompressor(w) (64).
  Always uses UNIFORM pooling for the hand-crafted stats — decoupled from compress
  routing; the two compressors additionally see the pre-pooling per-lead backbone
  features and the HuBERT-derived routing weights respectively.

  tau_detach (constructor flag, default True) controls whether ALL of the above is
  detached before it reaches the tau head:
    True  (default): tau is a read-out and not a lever. The NLL log(tau) + |e|/tau
           could otherwise be reduced by flattening attention / inflating tau
           instead of localizing better. Detached, the localizer only ever sees
           |e|/tau as a per-beat constant reweighting, and the compressors' own
           weights train off the tau NLL without ever handing gradient back to the
           backbone, scorer, or compress modules.
    False: nothing is detached — gradient from the tau NLL flows all the way back
           into the shared backbone and compress path, same entanglement v4's
           coef_head has (t_on/t_off/tau_on/tau_off as 4 channels of one conv, no
           detach anywhere). Exists as a controlled ablation against tau_detach=True
           to test whether v4's tau being systematically larger is a genuine scale
           difference or an artifact of that entanglement letting v4 cheat NLL by
           inflating tau instead of localizing better.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeadFeatureCompressor(nn.Module):
    """Summarizes pre-pooling per-lead backbone features into a fixed embedding.

    Input h_leads: (N, L, C, W) — the SAME per-lead conv output the scorer reads,
    before attention pooling collapses it. Mean+max pooled over time per lead,
    flattened across leads, then projected. Detachment (if any) happens at the
    call site, not in here, so this module is reusable for both tau_detach modes.
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
    vector: with only L=12 real numbers of signal, a hand-crafted feature is
    cheaper to trust than making the projection re-derive it on ~5k training
    beats. Detachment happens at the call site, same as LeadFeatureCompressor.
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
    FILM_CHANNELS = 32
    _C = 2.0 * math.log(19)

    def __init__(self, window_size=None, embed_dim=768, scale=1, scorer_scale=1,
                 tau_cap=None, tau_floor=1.0, tau_c_init_on=1.0, tau_c_init_off=10.0,
                 tau_detach=True, offset_prior=True, no_compress=False,
                 compress_mode='full', detach_routing=False, center_context=False, use_ste=False,
                 feature_pool=False, **kwargs):
        super().__init__()
        if window_size is None:
            from base.beat import WINDOW_PRE, WINDOW_POST
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
        if feature_pool and detach_routing:
            raise ValueError('--detach_routing is incompatible with --feature_pool: it would cut '
                             'the backbone off from the pooled loss, removing the contamination '
                             'path that feature pooling exists to create.')
        # int() so a fractional --scale still yields integer Conv1d channel
        # counts; identical to the old behaviour for integer scales.
        C  = int(self.FILM_CHANNELS * scale)
        SC = max(1, round(scorer_scale * C))

        # HuBERT compress: (N, D, L, t) → (N, 2, L)  [scalar onset/offset lead weights]
        # compress_mode controls capacity: 'full' (original), 'small' (tiny bottleneck),
        # 'avgpool' (mean-pool over time + linear — minimal params, no temporal memorization)
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
                # 768→4 bottleneck, 1 strided conv, then adaptive pool — ~3K params
                self.compress = nn.Sequential(
                    nn.Conv2d(embed_dim, 4, kernel_size=(1, 1)),
                    nn.GELU(),
                    nn.Conv2d(4, 4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d((None, 1)),
                    nn.Conv2d(4, 2, kernel_size=(1, 1)),
                )
            elif compress_mode == 'avgpool':
                # mean-pool over time, then linear projection — ~1.5K params, no temporal overfitting
                self.compress = nn.Sequential(
                    nn.AdaptiveAvgPool2d((None, 1)),
                    nn.Conv2d(embed_dim, 2, kernel_size=(1, 1)),
                )
            else:
                raise ValueError(f'Unknown compress_mode: {compress_mode!r}')

        # Per-lead backbone: shared weights, applied via reshape (N*L, 2, W) → (N*L, C, W)
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

        # Scorer: (M, C, W) → (M, W) — shared between PATH A and PATH B
        def _make_scorer():
            return nn.Sequential(
                nn.Conv1d(C,    SC*2, 7, padding=3, padding_mode='reflect'), nn.GELU(),
                nn.Conv1d(SC*2, SC,   3, padding=1, padding_mode='reflect'), nn.GELU(),
                nn.Conv1d(SC,   C//4, 3, padding=1, padding_mode='reflect'), nn.GELU(),
                nn.Conv1d(C//4, 1,    3, padding=1, padding_mode='reflect'),
            )

        self.onset_scorer  = _make_scorer()
        self.offset_scorer = _make_scorer()

        # Tau head: MLP on [5 hand-crafted entropy scalars, 256 lead-feature
        # embedding, 64 routing embedding] = 325 inputs, all subject to tau_detach.
        #
        # BatchNorm on the inputs because the components differ by orders of
        # magnitude: var_t/W^2 is ~3e-4 for a realistic inter-lead scatter while the
        # normalized entropies sit near 0.5, so without standardization a single
        # Kaiming-init layer cannot see var_t at all -- same reasoning now applies
        # to mixing the raw 5 scalars with the learned 256+64 embeddings.
        TAU_IN = 5 + 256 + 64

        def _make_tau_head():
            return nn.Sequential(
                nn.BatchNorm1d(TAU_IN),
                nn.Linear(TAU_IN, 64), nn.GELU(),
                nn.Linear(64, 1),
            )

        self.onset_tau_head  = _make_tau_head()
        self.offset_tau_head = _make_tau_head()

        self.lead_feat_compressor_on  = LeadFeatureCompressor(C)
        self.lead_feat_compressor_off = LeadFeatureCompressor(C)
        self.routing_compressor_on    = RoutingCompressor()
        self.routing_compressor_off   = RoutingCompressor()

        # tau = tau_c + tau_b: a per-target learned constant (the prior --
        # different init for onset vs offset, moves slowly) plus a per-beat
        # deviation from the tau head (starts at exactly 0 for every beat,
        # also slow -- see the optimizer param groups in train.py). This
        # replaces tau_floor as the source of tau's initial scale; tau_floor
        # is kept as a constructor arg for old recipes but unused here.
        self.tau_c_on  = nn.Parameter(torch.tensor(float(tau_c_init_on)))
        self.tau_c_off = nn.Parameter(torch.tensor(float(tau_c_init_off)))

        # Only the LAST weight is zeroed, bias set to exactly 0 so tau_b == 0
        # for every beat at t=0. Zeroing both (the previous version) is an
        # exact saddle the head can never leave: h = GELU(0) = 0 makes dL/dW2 = 0,
        # and W2 = 0 makes dL/dW1 = 0, so each zero block protects the other and
        # only the output bias ever trains. With W1 at its default init, h != 0,
        # so W2 gets gradient immediately and W1 unblocks as soon as W2 leaves
        # zero, while out == 0 still holds at t=0.
        for head in (self.onset_tau_head, self.offset_tau_head):
            nn.init.zeros_(head[3].weight)
            nn.init.zeros_(head[3].bias)

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
        """score: (M, W) → attn (M, W), t (M,)"""
        attn   = F.softmax(score, dim=-1)
        t_soft = (attn * self.t_grid).sum(-1)
        if self.use_ste:
            idx    = score.argmax(dim=-1)
            t_hard = self.t_grid[0, idx]
            t_soft = t_hard + (t_soft - t_soft.detach())
        return attn, t_soft

    def _maybe_detach(self, x):
        """tau_detach=True (default): cut gradient here. False: pass through
        unchanged, letting the tau NLL train the backbone/compress path too --
        the v4-style entangled ablation. One flag, applied at every call site
        that feeds the tau head, so nothing is hardcoded detached or attached."""
        return x.detach() if self.tau_detach else x

    def _tau_from_all(self, tau_head, lead_compressor, routing_compressor,
                       attn_per_lead, t_leads, h_leads, w, tau_c):
        """Beat-level tau from entropy stats + learned lead-feature and
        routing-weight embeddings.

        attn_per_lead: (N, L, W) — per-lead attention distributions
        t_leads:       (N, L)   — per-lead soft-argmax positions
        h_leads:       (N, L, C, W) — pre-pooling per-lead backbone features
        w:             (N, L)   — HuBERT-derived routing weights (w_on or w_off)
        tau_c:         scalar   — self.tau_c_on or self.tau_c_off
        Returns:       (N,) tau

        Uniform pooling is used for the hand-crafted stats regardless of
        compress — that part of tau stays decoupled from routing the same way
        it always has. Every input is passed through _maybe_detach right
        before use, so tau_detach controls the whole feature set uniformly.
        """
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

        feat    = torch.cat([feat_lead, feat_hub, hand_feat], dim=-1)  # (N,325)
        tau_b   = tau_head(feat)[:, 0]           # per-beat deviation, 0 at t=0
        tau     = (tau_c + tau_b).clamp(min=1.0)  # safety floor only -- tau_c carries the prior
        if self.tau_cap is not None:
            tau = tau.clamp(max=self.tau_cap)
        return tau

    def _forward_impl(self, window, context, routing_alpha=1.0):
        N, L, _, W = window.shape
        C = int(self.FILM_CHANNELS * self.scale)

        # ── backbone: shared weights, all leads batched ───────────────────────
        w          = window.reshape(N * L, 2, W)
        w          = torch.stack([w[:, 0], w[:, 1].abs()], dim=1)
        f          = self._norm(w)
        h_on_flat  = self._norm(self.onset_backbone(f))    # (N*L, C, W)
        h_off_flat = self._norm(self.offset_backbone(f))

        # ── scorer: per-lead attention maps ──────────────────────────────────
        score_on = self.onset_scorer(h_on_flat).squeeze(1)   # (N*L, W)
        attn_on_flat, t_on_flat = self._softargmax(score_on) # (N*L, W), (N*L,)

        # offset prior applied per lead using per-lead t_on
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

        # ── Routing (moved ahead of Tau): tau's compressors now consume w_on/w_off,
        # so this has to exist before the tau block runs. Nothing else about this
        # block changed from its old position further down. ─────────────────────
        if self.no_compress:
            w_on  = torch.full((N, L), 1.0 / L, device=f.device, dtype=f.dtype)
            w_off = w_on
            film_m = {
                'film_gamma_dev': torch.tensor(0.0),
                'film_beta_dev':  torch.tensor(0.0),
                'film_ratio':     torch.tensor(0.0),
            }
        else:
            # compress: (N, 768, L, 38) → (N, 2, L)
            # routing_alpha=0 → uniform prior; 1 → full routing
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

        # ── Tau: beat-level, from all-lead attention stats + lead-feature and
        # routing-weight embeddings. Computed once, shared by PATH A and PATH B.
        h_on_leads_flat  = h_on_flat.view(N, L, C, W)
        h_off_leads_flat = h_off_flat.view(N, L, C, W)
        tau_on  = self._tau_from_all(self.onset_tau_head,  self.lead_feat_compressor_on,  self.routing_compressor_on,
                                      attn_on_per_lead,  t_on_leads,  h_on_leads_flat,  w_on,  self.tau_c_on)   # (N,)
        tau_off = self._tau_from_all(self.offset_tau_head, self.lead_feat_compressor_off, self.routing_compressor_off,
                                      attn_off_per_lead, t_off_leads, h_off_leads_flat, w_off, self.tau_c_off)

        # ── PATH A: per-lead t with shared beat-level tau ─────────────────────
        coefs_a = torch.stack([
            t_on_leads,  tau_on.unsqueeze(-1).expand(N, L),
            t_off_leads, tau_off.unsqueeze(-1).expand(N, L),
        ], dim=-1)  # (N, L, 4)

        # ── pool + localize ──────────────────────────────────────────────────
        if self.feature_pool:
            # v4-style: pool the BACKBONE FEATURES, then score the fused map.
            # A corrupted lead contaminates h_fused irreversibly — the scorer has
            # no way to discard it, so the augmentation acts as input noise on the
            # localizer itself. Pooling attention instead lets a flat distribution
            # contribute only a bounded pull toward the window centre, which the
            # router can zero out — which is why noise only ever trained the router.
            #
            # PATH A is this same operation at the one-hot corners w = e_l: since
            # h_*_flat is already _norm'd and _norm is idempotent, w = e_l gives
            # hf == h_leads[:, l] exactly, and the scorer is shared. So the two NLL
            # terms are one function sampled at the 12 corners and at the HuBERT
            # interior point. detach_routing is therefore NOT applied here: cutting
            # the backbone off from the pooled loss would let it learn features that
            # only localize in isolation, never features that survive mixing — which
            # is the entire mechanism being tested.
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
            # soft-argmax on pooled attention
            t_on  = (pooled_attn_on  * self.t_grid).sum(-1)   # (N,)
            t_off = (pooled_attn_off * self.t_grid).sum(-1)

        coefs_b = torch.stack([t_on, tau_on, t_off, tau_off], dim=-1)   # (N, 4)

        mask      = self._coefs_to_mask(coefs_b[:, 0], coefs_b[:, 1], coefs_b[:, 2], coefs_b[:, 3])
        durations = (coefs_b[:, 2] - coefs_b[:, 0]).clamp(min=0).unsqueeze(-1)

        # per-lead backbone features for visualization
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
