# QRS delineation ensemble — integration bundle

Built from the **completed** training runs. Layout, schema, API and weights are
final; see Caveats for what is still outstanding.

## Files

| file | what |
|---|---|
| `qrs_model.py` | vendored `MaskHeadV6`. **Do not edit** — byte-identical to the training repo. |
| `qrs_ensemble.py` | vendored loader + reference `predict()`. **Do not edit**. |
| `qrs_ensemble_deferred.py` | one GPU→CPU sync instead of K (1.08x). Base of the below. |
| `qrs_ensemble_optimized.py` | **what the app runs**: deferred + `torch.compile` + fp16. |
| `package_ensemble.py` | converter: run directories → a `.pt` bundle. |
| `make_golden.py` / `regen_golden.py` | record / re-record the golden fixtures. |
| `beat.py` | beat detection + window extraction. |
| `dataset.py` | `preprocess_hubert` — window → encoder input. |
| `encoder.py` | the HuBERT-ECG encoder (fetched from the Hub, not bundled). |
| `recording.py` | the pipeline the app calls: `Recording.from_signal(..., predict=True)`. |
| `weights/` | the `.pt` bundles and `golden_*.npz` fixtures (~145MB). |

The layout is **flat on purpose**: the two vendored files import each other by
bare name, so `__init__.py` puts this directory on `sys.path` and a re-sync
from the training repo stays a straight file copy.

There is **one model**. The FiLM (v4) head, `MaskHeadV1`, the `PTHead` baseline,
the mid-training `v6_daint` checkpoint and the duplicate in-package `MaskHeadV6`
were all removed — they predicted from different, unvalidated weights, and a
silent fallback to them produced numbers no golden fixture covers. A missing
bundle now raises.

Both `.pt` files load through the same `.py`: their members are architecturally
identical (linear / no-HuBERT τ head, 336,341 tensor entries, window 550). Only
the member list and its provenance differ, so the two can be swapped, or merged
into a hybrid, without touching the host app.

## Use

```python
import os
from ecg_nn.production import load_optimized, BUNDLE_DIR
ens = load_optimized(os.path.join(BUNDLE_DIR, "megamodel_loo32.pt"), device="cuda")
out = ens.predict(window, emb)
# out["onset"], out["offset"], out["duration"], out["tau_on"], out["tau_off"],
# out["spread_on"], out["spread_off"]   -- all (N,) numpy, one row per beat
```

### Input contract

```
window : (N, L, 2, 550)   per-lead ECG window, 2 channels
emb    : (N, L,  38, 768) HuBERT embedding of that window
```
`L` = 12 leads. Both come from the existing beat-extraction + HuBERT pipeline;
this bundle does not reimplement them.

### Output

`onset` and `offset` are per-beat medians across members, taken independently.
`duration = median(offset) − median(onset)` — **not** the median of the members'
durations. Those are different statistics and the difference is not cosmetic:
boundary errors are correlated within a member, and medianing the difference
throws that structure away.

`spread_on` / `spread_off` are the cross-member standard deviations. They are
reported *beside* τ, never folded into it: members carry separately calibrated
scales, so a spread and a τ answer different questions. Which one to surface is
a product decision that is deliberately left open.

## HuBERT

Not bundled. It must be the **same checkpoint** the members were trained
against.

This is the one failure mode that produces no error. A different encoder yields
identical tensor shapes, so nothing raises — every member simply reads a feature
space it never saw, and they all degrade together, so the ensemble median hides
it rather than exposing it. Hence:

```python
ens.check_encoder(hubert.state_dict())   # True / False / None if no hash recorded
```

`manifest["encoder_sha256"]` is **empty in this mock** and must be filled before
any real deployment (`package_ensemble.py --encoder_sha256 …`).

## Precision on GPU

`QRSEnsemble` disables TF32 whenever it loads onto CUDA:

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

**Do not re-enable these after loading.** PyTorch turns TF32 on by default on
recent NVIDIA GPUs — a 10-bit mantissa against fp32's 23. In most networks that
is harmless; here it is not, because the model takes a soft-argmax over 550
timesteps, so small logit perturbations shift the weighted centroid.

Measured on this bundle, against the same inputs on CPU:

| | onset error |
|---|---|
| GPU, PyTorch defaults | **6.8 ms** |
| GPU, TF32 disabled | **0.006 ms** |

6.8 ms is comparable to the model's own onset MAE (8.47 ms), and it appears
silently — no warning, no exception, just worse predictions. Disabling TF32
costs nothing measurable (2.5 s vs 2.7 s for 32 members over 12 beats).

These are process-global flags. If the host application sets them for its own
reasons, it must do so *before* constructing `QRSEnsemble`, or the constructor's
setting will be overwritten.

## Golden test

```python
from ecg_nn.production import run_golden, run_golden_optimized
run_golden("megamodel_loo32.pt", "golden_megamodel_loo32.npz")            # reference fp32 path
run_golden_optimized("megamodel_loo32.pt", "golden_megamodel_loo32.npz")  # the path that ships
```

Run **both**: the first pins the weights and the architecture, the second pins
the fp16 boundary and the compiled kernels. `python qrs_ensemble_optimized.py`
runs the second over every bundle present.

Catches wrong weights, wrong architecture flags, changed kernel numerics, a
mis-shaped input contract and member-ordering bugs in the median — in one call.
Run it on install.

The fixture is **12 real beats, 3 from each of the four patients**, so it also
exercises realistic value ranges and produces outputs you can sanity-check by
eye. Three beats from each patient rather than twelve from one is for coverage:
four morphologies, four annotation styles.

It does **not** validate the HuBERT encoder or beat extraction: the inputs are
stored *post*-encoder. `check_encoder()` covers the encoder half; extraction is
uncovered and would need a fixture stored pre-encoder.

Outputs are recorded on CPU, but replay on GPU is fine provided TF32 stays
disabled (see above). The tolerance is **0.5 ms**, chosen to sit far above
device noise and far below any real failure: with TF32 off, GPU and CPU agree
to 0.006 ms on the boundaries and 0.09 ms on τ, while a genuinely wrong set of
weights moved the recorded durations by 144 ms — roughly 300× the tolerance.

## Caveats

- **Provenance of the weights.** `megamodel_loo32` is the four LOO folds of
  `v6_lin_nohub_s2at1000` (phase-2 switch at epoch 1000, 1500 total).
  `allseed64` is `v6_ship_all` at epoch 3000.
- **`allseed64` has no honest validation and cannot get one.** Every patient is
  in its training set, so nothing is held out. Its accuracy can only be
  estimated from the LOO folds of the *other* build, or measured on a genuinely
  new patient.
- **Provenance matters.** `held_out` is the patient a member never saw
  (`None` for all-patient members). At inference the two kinds are
  interchangeable; for evaluation they are not, and this field is the only
  record of which is which.
- **`encoder_sha256` is still empty** and must be filled before deployment;
  until then `check_encoder()` returns `None` rather than verifying anything.

## Performance

`qrs_ensemble_optimized.py` is what `Recording._run_inference` builds. It is the
deferred-sync loop plus `torch.compile(mode='max-autotune')` plus fp16 on every
submodule the golden fixture tolerates. Measured on an RTX 4070 Ti SUPER
(torch 2.6.0+cu124, `megamodel_loo32`, K=32, B=32):

| path | B=32 | per beat | |
|---|---|---|---|
| plain loop (`QRSEnsemble.predict`) | ~122 ms | 3.81 ms | 1.00x |
| + deferred sync | 122.0 ms | 3.81 ms | 1.00x |
| + `torch.compile` | 66.4 ms | 2.08 ms | 1.84x |
| **+ fp16-safe modules (ships)** | **58.4 ms** | **1.83 ms** | **2.09x** |

Worst golden deviation for that config: **0.130 ms** (offset), against the
0.5 ms tolerance. Roughly 3.8 s → 1.8 s of head inference per ~1000-beat
recording.

`torch.compile` needs inductor's Triton backend, which requires **CUDA compute
capability >= 7.0** (Turing+). On older cards — the local GTX 1080 is 6.1 — it is
skipped automatically and the path runs deferred + fp16 only (~1.12x). Check
`.compiled` / `.fp16` on the returned object for what actually happened.

**fp16 stops at the backbones**, which stay fp32 — casting either one moves
onset by ~23 ms, 46× outside tolerance, and casting only its last layer is just
as bad. Not overflow (activations shrink 120 → 0.16), not subnormal underflow
(≤1.4% of values below fp16's smallest normal), and bf16 is worse still. The
boundary is empirical; `FP16_SAFE` in `qrs_ensemble_optimized.py` is the list,
and its `main()` re-checks it against the fixtures.

The path is memory-bandwidth-bound, not FLOP-bound: ~11 GFLOP/beat at K=32
against ~11 GB of activations, i.e. ~7 FLOP/byte where the card's balance is
~48. That is why fusion paid and why launch-count optimizations did not — see
`dev/attic/README.md` for the three that were tried and measured.
