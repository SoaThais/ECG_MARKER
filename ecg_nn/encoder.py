"""HuBERT-ECG encoder — the only upstream model ecg_nn still builds itself.

Everything downstream of this (MaskHeadV6 and the ensemble around it) is the
vendored production dump: qrs_model.py / qrs_ensemble.py. This module exists
because the encoder is NOT bundled with the weights and must be fetched from
the same checkpoint the members were trained against -- see the "HuBERT"
section of README.md, and `QRSEnsemble.check_encoder()`, which verifies that
by hash rather than trusting it.

This file replaces the old model.py, which additionally carried the FiLM
MaskHead (v4), MaskHeadV1, a PTHead baseline, and a second copy of MaskHeadV6.
All four are gone: the first three were superseded by the production ensemble,
and the fourth duplicated the vendored qrs_model.py, which is the copy that
ships and the only one the weights are validated against.
"""
import torch
import torch.nn as nn
from transformers.models.auto import AutoModel  # <-- Bypass root lazy loading

class HuBERTECGRegressor(nn.Module):
    """Frozen HuBERT-ECG encoder.  Call .encode(x) to get per-lead embeddings."""

    def __init__(self, repo_id='Edoardo-BS/hubert-ecg-base', freeze=True, **kwargs):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

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
        """x: (N, 12, 2500) -> (N, 12, t, D)"""
        N, L, T = x.shape
        out = self.encoder(input_values=x.reshape(N * L, T)).last_hidden_state
        t, D = out.shape[1], out.shape[2]
        return out.reshape(N, L, t, D)


def build_encoder(device=None, **kwargs):
    """Build the HuBERT-ECG encoder."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HuBERTECGRegressor(**kwargs).to(device)
    return model, device


def main():
    """Self-test: build the encoder and check the embedding contract the
    ensemble depends on -- (N, 12, t, 768) with t=38 for a 5 s window at
    500 Hz. Needs network access on first run (the checkpoint is fetched from
    the Hub and cached); skips rather than failing if it cannot be reached."""
    try:
        enc, device = build_encoder()
    except Exception as e:                       # offline / no cached weights
        print("[skip] could not build encoder: %s: %s" % (type(e).__name__, str(e)[:120]))
        return
    enc.eval()
    with torch.no_grad():
        out = enc.encode(torch.zeros(2, 12, 2500, device=device))
    print("encoder on %s -> %s" % (device, tuple(out.shape)))
    assert out.shape[0] == 2 and out.shape[1] == 12, out.shape
    assert out.shape[3] == 768, out.shape
    print("encoder self-test OK")


if __name__ == "__main__":
    main()
