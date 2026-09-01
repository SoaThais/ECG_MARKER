# QRS ensemble — weights

Weights and fixtures only. **The code moved to `ecg_nn/production/`** (vendored
`MaskHeadV6` + loader, the bundle converter, and the optimized inference path
the app actually runs); see `ecg_nn/production/README.md`.

| file | what |
|---|---|
| `megamodel_loo32.pt` | 32 members, 4 LOO folds × 8 seeds. The default (`4fold`, and `light` subsets it). |
| `allseed64.pt` | 64 members, all patients × 64 seeds (`complete`). |
| `golden_megamodel_loo32.npz` | 12 real beats (3 per patient) + recorded outputs. |
| `golden_allseed64.npz` | same, for the 64-member bundle. |

They stay out of the package because they are ~145MB together; `ecg_nn.production`
exposes this directory as `BUNDLE_DIR`, and `Recording._run_inference` reads
them from here.

```python
import os
from ecg_nn.production import load_optimized, BUNDLE_DIR
ens = load_optimized(os.path.join(BUNDLE_DIR, "megamodel_loo32.pt"))
```
