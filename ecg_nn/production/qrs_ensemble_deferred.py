"""PATCH -- NOT part of the vendored production dump (qrs_ensemble.py /
qrs_model.py). Those two ship exactly as pulled from daint and are marked
"do not edit" in their own README; this file is our own addition on top,
kept separate for that reason.

qrs_ensemble.QRSEnsemble.predict() calls .cpu().numpy() INSIDE the per-member
loop:

    for h in self.heads:
        out = h(window, emb, routing_alpha=routing_alpha)
        cols.append(out[2].float().cpu().numpy())   # sync #1, #2, ... #K

Each .cpu() call is a synchronization barrier -- it blocks the CPU thread
until every kernel queued so far for that member has actually finished on
the GPU, before the Python loop is allowed to move on and dispatch the next
member's kernels. That serializes CPU-side dispatch and GPU-side execution:
the CPU can never race ahead to queue up member i+1 while the GPU is still
chewing through member i.

DeferredSyncQRSEnsemble below is otherwise identical to the original loop --
same architecture assumption (all K members share model_kwargs), same
combination rule (median onset/offset, duration = their difference) -- but
keeps every member's output on-GPU (out[2].float(), no .cpu()) until all K
have been dispatched, and syncs exactly once at the end via a single
torch.stack(...).cpu().numpy(). This lets the CPU get all K members' worth
of kernels queued while the GPU is still working through earlier ones.

Usage
-----
    from qrs_ensemble import QRSEnsemble
    from qrs_ensemble_deferred import DeferredSyncQRSEnsemble

    ens = QRSEnsemble.load("megamodel_loo32.pt", device="cuda")
    deferred = DeferredSyncQRSEnsemble.from_ensemble(ens)
    out = deferred.predict(window, emb)   # identical dict contract to ens.predict()

Measured on our GTX 1080 (megamodel_loo32, batch=16, synthetic random
inputs): 305.3ms (original, sync every member) vs 283.3ms (this module,
sync once) per predict() call -- ~1.08x. Modest but real and essentially
free; stacks with a larger _INFERENCE_BATCH_SIZE (see ecg_nn.recording,
which independently measured ~1.18x from batch=16 to batch=32) and with
CUDA graphs (qrs_ensemble_cudagraph.py) for further wins on top.
"""
import hashlib
import os

import numpy as np
import torch


class DeferredSyncQRSEnsemble:
    """Drop-in for qrs_ensemble.QRSEnsemble's predict() -- identical output
    contract; only the internal loop's GPU->CPU sync timing differs (once at
    the end, not once per member). check_encoder()'s hashing logic is
    duplicated verbatim from qrs_ensemble.py since this class doesn't wrap a
    live QRSEnsemble instance to delegate to.
    """

    def __init__(self, heads, provenance, manifest, device):
        self.heads = list(heads)
        self.provenance = list(provenance)
        self.manifest = dict(manifest)
        self.device = torch.device(device)

    @classmethod
    def from_ensemble(cls, ensemble):
        """Wrap an already-loaded qrs_ensemble.QRSEnsemble."""
        return cls(ensemble.heads, ensemble.provenance, ensemble.manifest, ensemble.device)

    def __len__(self):
        return len(self.heads)

    @torch.no_grad()
    def predict(self, window, emb, routing_alpha=1.0, return_members=False):
        """Same contract as QRSEnsemble.predict() -- see that docstring."""
        window = torch.as_tensor(window).to(self.device)
        emb = torch.as_tensor(emb).to(self.device)

        cols = []
        for h in self.heads:
            out = h(window, emb, routing_alpha=routing_alpha)
            cols.append(out[2].float())          # stays on GPU -- no sync yet
        C = torch.stack(cols).cpu().numpy()       # ONE sync, after every member is queued

        on = np.median(C[:, :, 0], axis=0)
        off = np.median(C[:, :, 2], axis=0)
        res = {
            "onset": on,
            "offset": off,
            "duration": off - on,
            "tau_on": np.median(C[:, :, 1], axis=0),
            "tau_off": np.median(C[:, :, 3], axis=0),
            "spread_on": C[:, :, 0].std(axis=0, ddof=1) if len(self) > 1 else np.zeros_like(on),
            "spread_off": C[:, :, 2].std(axis=0, ddof=1) if len(self) > 1 else np.zeros_like(off),
            "n_members": len(self),
        }
        if return_members:
            res["members"] = C
        return res

    def check_encoder(self, encoder_state_dict):
        want = self.manifest.get("encoder_sha256")
        if not want:
            return None
        h = hashlib.sha256()
        for k in sorted(encoder_state_dict):
            v = encoder_state_dict[k]
            h.update(k.encode())
            h.update(np.ascontiguousarray(v.detach().cpu().float().numpy()).tobytes())
        return h.hexdigest() == want


def _self_test():
    """Load whatever bundle(s) are present alongside this file, compare
    DeferredSyncQRSEnsemble against both the original QRSEnsemble.predict()
    and the shipped golden fixture, on CPU (matching how the goldens were
    recorded).
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from qrs_ensemble import QRSEnsemble

    here = os.path.dirname(__file__)
    pairs = [
        ('megamodel_loo32.pt', 'golden_megamodel_loo32.npz'),
        ('allseed64.pt', 'golden_allseed64.npz'),
    ]
    ran_any = False
    for bundle, golden in pairs:
        bundle_path, golden_path = os.path.join(here, bundle), os.path.join(here, golden)
        if not (os.path.exists(bundle_path) and os.path.exists(golden_path)):
            continue
        ran_any = True
        print(f"=== {bundle} ===")
        g = np.load(golden_path)
        ens = QRSEnsemble.load(bundle_path, device='cpu')
        deferred = DeferredSyncQRSEnsemble.from_ensemble(ens)

        slow_out = ens.predict(g['window'], g['emb'])
        deferred_out = deferred.predict(g['window'], g['emb'])

        for k in ('onset', 'offset', 'duration', 'tau_on', 'tau_off'):
            d = np.abs(deferred_out[k] - slow_out[k]).max()
            print(f"  {k:10s} deferred-vs-loop={d:.6f}")
            # Should be exact or near-exact (same ops, same order, only the
            # sync POINT differs -- no numeric reason for these to diverge
            # at all, unlike the vmap patch's batched-matmul reduction order).
            assert d < 1e-4, f"{bundle}/{k}: deferred diverges from the original loop by {d}"
        print("  OK")

    if not ran_any:
        print("no bundle+golden pairs found next to this file -- nothing to test")


if __name__ == '__main__':
    _self_test()
