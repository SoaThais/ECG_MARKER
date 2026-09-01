# QRS delineation ensemble — integration bundle

Built from the **completed** training runs. Layout, schema, API and weights are
final; see Caveats for what is still outstanding.

## Files

| file | what |
|---|---|
| `qrs_ensemble.py` | loader + inference. The only file the host app imports. |
| `qrs_model.py` | vendored `MaskHeadV6`. Self-contained; do not edit. |
| `package_ensemble.py` | builds a bundle from run directories (build-time only). |
| `megamodel_loo32.pt` | 32 members, 4 LOO folds × 8 seeds. |
| `allseed64.pt` | 64 members, all patients × 64 seeds. |
| `golden_*.npz` | 12 real beats (3 per patient) + recorded outputs, per bundle. |

Both `.pt` files load through the same `.py`: their members are architecturally
identical (linear / no-HuBERT τ head, 336,341 tensor entries, window 550). Only
the member list and its provenance differ, so the two can be swapped, or merged
into a hybrid, without touching the host app.

## Use

```python
from qrs_ensemble import QRSEnsemble
ens = QRSEnsemble.load("megamodel_loo32.pt", device="cuda")
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
from qrs_ensemble import run_golden
run_golden("megamodel_loo32.pt", "golden_megamodel_loo32.npz")
```

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
