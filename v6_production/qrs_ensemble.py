"""Shipping inference for the QRS delineation ensemble.

One .py, one .pt. Both deliverables -- the LOO megamodel and the all-patient
multi-seed build -- use this same loader, because their members are
architecturally identical (linear / no-HuBERT tau head, 331,606 params); only
the member list and its provenance differ.

    from qrs_ensemble import QRSEnsemble
    ens = QRSEnsemble.load('megamodel_loo32.pt', device='cuda')
    out = ens.predict(window, emb)      # -> dict of numpy arrays, one row/beat

`window` and `emb` are exactly what the training pipeline feeds the model:
    window : (N, L, 2, W)   per-lead ECG window (2 channels)
    emb    : (N, L, T, 768) HuBERT embedding for that window
The HuBERT encoder is NOT bundled in this file -- it must be the same
checkpoint the members were trained against. See `encoder` in the manifest and
check_encoder() below: a different encoder produces the same tensor SHAPES and
no error, while every member silently reads a feature space it never saw, so
this is verified by hash rather than trusted.

Combination is a per-beat MEDIAN across members, taken independently for onset
and offset. Duration is then median(offset) - median(onset) -- NOT the median of
the members' durations, which is a different statistic and worse: the boundary
errors are correlated within a member, and medianing the difference discards
that structure.
"""
import hashlib
import os

import numpy as np
import torch

from qrs_model import MaskHeadV6

SCHEMA = 1


class QRSEnsemble:
    def __init__(self, members, model_kwargs, provenance, manifest, device="cpu"):
        self.model_kwargs = dict(model_kwargs)
        self.provenance = list(provenance)
        self.manifest = dict(manifest)
        self.device = torch.device(device)
        if self.device.type == "cuda":
            # PyTorch enables TF32 by default on recent NVIDIA GPUs: a 10-bit
            # mantissa against fp32's 23. Measured effect on this model is 6.8 ms
            # of onset error -- comparable to the model's own 8.47 ms MAE, and
            # entirely silent. The soft-argmax over 550 timesteps amplifies small
            # logit perturbations, so reduced-precision matmul is not tolerable
            # here even though it is harmless in most networks. Disabling it
            # brings GPU within 0.006 ms of CPU at no measurable speed cost.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.heads = []
        for sd in members:
            h = MaskHeadV6(**self.model_kwargs).to(self.device)
            h.load_state_dict(sd)
            h.eval()          # required: the tau head opens with BatchNorm1d,
                              # which raises on a batch of 1 in train mode.
            self.heads.append(h)

    # ---------------------------------------------------------------- load
    @classmethod
    def load(cls, path, device="cpu"):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("schema") != SCHEMA:
            raise ValueError("unsupported bundle schema %r (expected %d)"
                             % (blob.get("schema"), SCHEMA))
        return cls(blob["members"], blob["model_kwargs"], blob["provenance"],
                   blob.get("manifest", {}), device=device)

    def __len__(self):
        return len(self.heads)

    # ------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, window, emb, routing_alpha=1.0, return_members=False):
        """-> dict with onset/offset/duration (ms) and tau, one row per beat."""
        window = torch.as_tensor(window).to(self.device)
        emb = torch.as_tensor(emb).to(self.device)
        cols = []
        for h in self.heads:
            out = h(window, emb, routing_alpha=routing_alpha)
            cols.append(out[2].float().cpu().numpy())   # (N,4)
        C = np.stack(cols)                              # (K, N, 4)

        on = np.median(C[:, :, 0], axis=0)
        off = np.median(C[:, :, 2], axis=0)
        res = {
            "onset": on,
            "offset": off,
            # median-onset / median-offset combination -- see module docstring
            "duration": off - on,
            "tau_on": np.median(C[:, :, 1], axis=0),
            "tau_off": np.median(C[:, :, 3], axis=0),
            # Cross-member spread. Reported, not folded into tau: members carry
            # separately-calibrated scales, so a spread and a tau are different
            # quantities and the caller should decide which to surface.
            "spread_on": C[:, :, 0].std(axis=0, ddof=1) if len(self) > 1 else np.zeros_like(on),
            "spread_off": C[:, :, 2].std(axis=0, ddof=1) if len(self) > 1 else np.zeros_like(off),
            "n_members": len(self),
        }
        if return_members:
            res["members"] = C
        return res

    # ------------------------------------------------------------ integrity
    def check_encoder(self, encoder_state_dict):
        """True when the supplied HuBERT matches the one the members expect.

        A mismatched encoder is the one failure that produces no error at all:
        shapes agree, and every member degrades together on a feature space it
        was not trained on. Verify, do not assume.
        """
        want = self.manifest.get("encoder_sha256")
        if not want:
            return None
        return _sha_state_dict(encoder_state_dict) == want


def _sha_state_dict(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k]
        h.update(k.encode())
        h.update(np.ascontiguousarray(
            v.detach().cpu().float().numpy()).tobytes())
    return h.hexdigest()


def run_golden(bundle_path, golden_path, atol=0.5):
    """Replay the recorded inputs and compare against recorded outputs.

    Catches, in one check: wrong weights, wrong architecture flags, changed
    kernel numerics, and a mis-shaped input contract. It does NOT validate the
    upstream beat extraction or the HuBERT encoder, because the golden inputs
    are stored post-encoder -- check_encoder() covers that half.

    atol=0.5 ms is chosen to sit far above device noise and far below any real
    failure. With TF32 disabled (see __init__) GPU and CPU agree to 0.006 ms on
    the boundaries and 0.09 ms on tau, while a genuinely wrong set of weights
    moved the recorded durations by 144 ms. A tolerance near machine epsilon
    would only produce spurious failures when the fixture is replayed on a
    different device from the one that recorded it.
    """
    g = np.load(golden_path)
    ens = QRSEnsemble.load(bundle_path)
    out = ens.predict(g["window"], g["emb"])
    bad = []
    for k in ("onset", "offset", "duration", "tau_on", "tau_off"):
        d = np.abs(out[k] - g[k]).max()
        if d > atol:
            bad.append("%s max|diff|=%.3g" % (k, d))
    if bad:
        raise AssertionError("golden mismatch: " + "; ".join(bad))
    return True


def main():
    """Self-test on a synthetic 2-member bundle: exercises save -> load ->
    predict and pins the two properties that are easy to get wrong."""
    import tempfile
    torch.manual_seed(0)
    kw = dict(window_size=64, embed_dim=768, scale=1, scorer_scale=0.5,
              tau_c_init_on=15.0, tau_c_init_off=15.0, tau_detach=False,
              offset_prior=False, compress_mode="full",
              tau_arch="linear", tau_feats="no_hubert")
    with tempfile.TemporaryDirectory() as tmp:
        members = [MaskHeadV6(**kw).state_dict() for _ in range(2)]
        p = os.path.join(tmp, "b.pt")
        torch.save({"schema": SCHEMA, "model_kwargs": kw, "members": members,
                    "provenance": [{"run": "synthetic", "held_out": None,
                                    "seed": i, "epoch": 0} for i in range(2)],
                    "manifest": {"combine": "median"}}, p)

        ens = QRSEnsemble.load(p)
        assert len(ens) == 2
        N, L, W = 3, 12, 64
        window = torch.randn(N, L, 2, W)   # 2 channels, not 1
        emb = torch.randn(N, L, 38, 768)
        out = ens.predict(window, emb, return_members=True)

        for k in ("onset", "offset", "duration", "tau_on", "tau_off"):
            assert out[k].shape == (N,), (k, out[k].shape)
        # duration must be the difference OF THE MEDIANS, not the median of the
        # differences -- the whole point of the combination rule.
        C = out["members"]
        assert np.allclose(out["duration"],
                           np.median(C[:, :, 2], 0) - np.median(C[:, :, 0], 0))
        med_of_diff = np.median(C[:, :, 2] - C[:, :, 0], axis=0)
        assert out["duration"].shape == med_of_diff.shape
        # tau must come from the model, never be derived from member spread
        assert not np.allclose(out["tau_on"], out["spread_on"])

        g = os.path.join(tmp, "g.npz")
        np.savez(g, window=window.numpy(), emb=emb.numpy(),
                 **{k: out[k] for k in ("onset", "offset", "duration",
                                        "tau_on", "tau_off")})
        assert run_golden(p, g)
    print("qrs_ensemble self-test OK")


if __name__ == "__main__":
    main()
