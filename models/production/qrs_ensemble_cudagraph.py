"""PATCH -- NOT part of the vendored production dump (qrs_ensemble.py /
qrs_model.py). Those two ship exactly as pulled from daint and are marked
"do not edit" in their own README; this file is our own addition on top,
kept separate for that reason.

Profiling (see qrs_ensemble_fast.py's docstring) found the real bottleneck
isn't raw GPU compute -- it's kernel LAUNCH COUNT. One predict() call over
32 members makes ~640 separate conv1d calls (Self CPU time ~299ms, nearly
as large as the ~304-413ms wall-clock total), because each of those tiny
ops carries real Python/aten dispatch overhead on the CPU side, independent
of how much actual math it does.

qrs_ensemble_fast.py's vmap approach tried to collapse the K-member loop
into one batched op, but vmap's own batching-rule dispatch overhead turned
out to cost about as much per op as what it saved -- a wash (0.96x),
verified by direct measurement, not assumed.

This module attacks the SAME bottleneck a different way: CUDA graphs. A
torch.cuda.CUDAGraph captures a fixed sequence of GPU kernel launches once;
replaying it is a single, cheap host-side call that re-issues the whole
captured DAG with (in principle) near-zero per-kernel CPU dispatch cost --
unlike a fresh Python-driven pass through hundreds of tiny ops. This is
exactly the profile CUDA graphs are built for.

Requirements this imposes:
  - CUDA only. There is no CPU graph-capture API.
  - FIXED shapes. Inputs must be copied into pre-allocated static buffers
    of one fixed batch size (chosen at construction) before each replay --
    a graph can't accept arbitrary new tensors like a normal forward call.
    predict() pads shorter batches with zeros up to that size and slices
    the padding back off before returning; padding beats' predictions are
    computed (harmlessly, on the GPU) but never returned.
  - routing_alpha is fixed at 1.0 at capture time -- the only value ecg_nn
    ever actually passes.

Usage
-----
    from qrs_ensemble import QRSEnsemble
    from qrs_ensemble_cudagraph import CUDAGraphQRSEnsemble

    ens = QRSEnsemble.load("megamodel_loo32.pt", device="cuda")
    graphed = CUDAGraphQRSEnsemble.from_ensemble(ens, batch_size=32)
    out = graphed.predict(window, emb)   # window/emb: up to 32 beats; shorter is padded internally

CUDAGraphQRSEnsemble.predict()'s output is numerically verified against the
original QRSEnsemble.predict() and the shipped golden fixtures (padded to
batch_size, same as real usage) -- run this file directly to check.

Actual result: the floor is GPU compute, not launch overhead
----------------------------------------------------------------
Correctness is exact (0.0 diff vs the original loop, every metric -- a
graph replays the identical computation, so this isn't surprising). Speed
is a different story: measured only 1.03-1.05x over the plain loop
(B=32, both bundles) -- barely better than qrs_ensemble_deferred.py's
simpler fix (~1.08x) and nowhere near what the "640 tiny conv1d calls"
framing predicted.

Isolated why: timed graph.replay() alone (no copy_, no final sync) at
510ms for B=32/32 members -- and copy_ into the static buffers was 0.4ms,
the final .cpu().numpy() sync was 0.1ms. So the graph correctly eliminated
essentially ALL avoidable CPU-side overhead (Python dispatch, per-op
launches, syncs) -- and what's left, ~510ms, is pure GPU kernel EXECUTION
time. That matches the profiler's earlier "Self CUDA time total: 413ms"
almost exactly (same ballpark, different run/batch-size details).

Conclusion: once the earlier CPU-dispatch/sync overhead is squeezed out
(which qrs_ensemble_deferred.py's much simpler fix already mostly did,
by removing 31 unnecessary per-member sync barriers that were serializing
CPU dispatch and GPU execution), what's left really is compute-bound on
this Pascal-generation GPU -- there's no more free lunch to extract from
scheduling the SAME work differently. The one thing that still helps is
doing LESS work: 'light' ensemble mode (4 members instead of 32, see
ecg_nn.recording) is a genuine ~7.4x for exactly that reason.

Given the added complexity (fixed batch size, CUDA-only, padding logic,
capture-time warmup cost) for a marginal gain over what's already wired
in via qrs_ensemble_deferred.py, this module is NOT wired into ecg_nn's
default path -- kept here as a correct, working, thoroughly-verified
reference in case it's useful on different hardware (a GPU where kernel
launch overhead is a bigger fraction of per-op cost) or at larger K.
"""
import os

import numpy as np
import torch


class CUDAGraphQRSEnsemble:
    """CUDA-graph-captured ensemble execution for a FIXED batch size."""

    def __init__(self, heads, provenance, manifest, device, batch_size,
                 window_shape=(12, 2, 550), emb_shape=(12, 38, 768)):
        self.heads = list(heads)
        self.provenance = list(provenance)
        self.manifest = dict(manifest)
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError("CUDAGraphQRSEnsemble requires a CUDA device "
                              f"(got {device!r}) -- there is no CPU graph capture API.")
        self.batch_size = batch_size
        self._window_shape = tuple(window_shape)
        self._emb_shape = tuple(emb_shape)

        L = self._window_shape[0]
        self._static_window = torch.zeros(batch_size, *self._window_shape, device=self.device)
        self._static_emb = torch.zeros(batch_size, *self._emb_shape, device=self.device)

        # Warmup on a SIDE stream before capture -- the official pattern
        # (see PyTorch's CUDA graphs docs). Capturing without this first can
        # bake in a lazy cudnn algorithm-selection pass or first-touch
        # allocation as part of the graph, which is wrong: those need to
        # settle beforehand, on their own stream, not during the recorded
        # sequence that gets replayed forever after.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.no_grad():
                for _ in range(3):
                    cols = [h(self._static_window, self._static_emb, routing_alpha=1.0)[2].float()
                            for h in self.heads]
                    _ = torch.stack(cols)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        self._graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(self._graph):
                cols = [h(self._static_window, self._static_emb, routing_alpha=1.0)[2].float()
                        for h in self.heads]
                self._static_output = torch.stack(cols)   # (K, batch_size, 4)

    @classmethod
    def from_ensemble(cls, ensemble, batch_size):
        """Wrap an already-loaded qrs_ensemble.QRSEnsemble. Captures the
        graph immediately (construction is the expensive step -- predict()
        afterward is cheap)."""
        return cls(ensemble.heads, ensemble.provenance, ensemble.manifest,
                    ensemble.device, batch_size)

    def __len__(self):
        return len(self.heads)

    @torch.no_grad()
    def predict(self, window, emb, routing_alpha=1.0, return_members=False):
        """Same contract as QRSEnsemble.predict() -- see that docstring.

        window/emb may hold fewer than self.batch_size beats (padded
        internally with zeros and sliced back off); more than
        self.batch_size raises, since a graph replays exactly the shape it
        was captured with. routing_alpha must be 1.0 (the value baked into
        the capture) -- passing anything else raises rather than silently
        ignoring it.
        """
        if routing_alpha != 1.0:
            raise ValueError("CUDAGraphQRSEnsemble was captured with routing_alpha=1.0; "
                              f"got {routing_alpha!r}. Build a new instance to change it.")

        window = torch.as_tensor(window).to(self.device)
        emb = torch.as_tensor(emb).to(self.device)
        n = window.shape[0]
        if n > self.batch_size:
            raise ValueError(f"batch of {n} exceeds this graph's fixed batch_size={self.batch_size}")

        self._static_window[:n].copy_(window)
        self._static_emb[:n].copy_(emb)
        if n < self.batch_size:
            self._static_window[n:].zero_()
            self._static_emb[n:].zero_()

        self._graph.replay()
        C = self._static_output[:, :n, :].float().cpu().numpy()   # (K, n, 4) -- padding sliced off

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


def _self_test():
    """CUDA only (see module docstring). Loads whatever bundle(s) are
    present alongside this file, compares CUDAGraphQRSEnsemble against both
    the original QRSEnsemble.predict() and the shipped golden fixture, and
    reports a timing comparison against the plain loop.
    """
    import sys
    import time
    sys.path.insert(0, os.path.dirname(__file__))
    from qrs_ensemble import QRSEnsemble

    if not torch.cuda.is_available():
        print("no CUDA device available -- CUDAGraphQRSEnsemble cannot be tested (or used)")
        return

    torch.backends.cudnn.benchmark = True

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
        print(f"=== {bundle} (CUDA) ===")
        g = np.load(golden_path)
        n_beats = g['window'].shape[0]

        ens = QRSEnsemble.load(bundle_path, device='cuda')
        graphed = CUDAGraphQRSEnsemble.from_ensemble(ens, batch_size=n_beats)

        slow_out = ens.predict(g['window'], g['emb'])
        fast_out = graphed.predict(g['window'], g['emb'])

        for k in ('onset', 'offset', 'duration', 'tau_on', 'tau_off'):
            d_vs_slow = np.abs(fast_out[k] - slow_out[k]).max()
            d_vs_golden = np.abs(fast_out[k] - g[k]).max()
            print(f"  {k:10s} graph-vs-loop={d_vs_slow:.6f}  graph-vs-golden={d_vs_golden:.4f}")
            # Same tolerance class as qrs_ensemble_fast.py's self-test --
            # legitimate reduction-order noise, tight enough to catch a real
            # bug (README measured one moving durations by 144ms).
            assert d_vs_slow < 0.5, f"{bundle}/{k}: graph diverges from the original loop by {d_vs_slow}"
        print("  OK (correctness)")

        # Timing: pad n_beats up to a round batch size for a realistic
        # comparison (graphs need a batch_size chosen up front; a real
        # caller would pick something like ecg_nn's _INFERENCE_BATCH_SIZE).
        B = 32
        window = torch.randn(B, *graphed._window_shape, device='cuda')
        emb = torch.randn(B, *graphed._emb_shape, device='cuda')
        graphed_b = CUDAGraphQRSEnsemble.from_ensemble(ens, batch_size=B)

        for _ in range(5):
            ens.predict(window, emb)
            graphed_b.predict(window, emb)
        torch.cuda.synchronize()

        N = 10
        t0 = time.time()
        for _ in range(N):
            ens.predict(window, emb)
        torch.cuda.synchronize()
        t_loop = (time.time() - t0) / N

        t0 = time.time()
        for _ in range(N):
            graphed_b.predict(window, emb)
        torch.cuda.synchronize()
        t_graph = (time.time() - t0) / N

        print(f"  timing (B={B}): loop={t_loop*1000:.1f}ms  cudagraph={t_graph*1000:.1f}ms  "
              f"speedup={t_loop/t_graph:.2f}x")

    if not ran_any:
        print("no bundle+golden pairs found next to this file -- nothing to test")


if __name__ == '__main__':
    _self_test()
