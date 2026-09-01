"""Build the golden fixture for every bundle: 3 real beats from each patient.

One file per bundle, 12 beats, shipped with it. Real beats rather than synthetic
tensors so the fixture also exercises realistic value ranges and yields outputs
that can be sanity-checked by eye -- a synthetic fixture pins reproducibility
but its predictions are meaningless, so nothing about it can be read.

Sampling 3 beats from each of the 4 patients rather than 12 from one is for
coverage: four morphologies, four annotation styles.

What it catches: wrong weights, wrong architecture flags, changed kernel
numerics, a broken input contract, member-ordering bugs in the median.
What it does not: the HuBERT encoder (inputs are stored post-encoder --
QRSEnsemble.check_encoder covers that) and beat extraction upstream of it.
"""
import glob
import re

import numpy as np
import torch

from qrs_ensemble import QRSEnsemble, run_golden

BUNDLES = ("megamodel_loo32", "allseed64")
KEYS = ("onset", "offset", "duration", "tau_on", "tau_off")

# ---- assemble the shared input set once ---------------------------------
ws, es, pats = [], [], []
for f in sorted(glob.glob("_g_p*.npz")):
    p = re.search(r"_g_(p\d+)\.npz", f).group(1)
    z = np.load(f)
    ws.append(z["window"])
    es.append(z["emb"])
    pats += [p] * len(z["window"])
window = np.concatenate(ws)
emb = np.concatenate(es)

# The loader hands the model (N, L, 2, W); the dumped windows are (N, L, W).
w = torch.as_tensor(window).float()
if w.dim() == 3:
    w = w.unsqueeze(2).repeat(1, 1, 2, 1)

for name in BUNDLES:
    ens = QRSEnsemble.load(name + ".pt")
    out = ens.predict(w, torch.as_tensor(emb).float())
    gp = "golden_%s.npz" % name
    np.savez(gp, window=w.numpy(), emb=emb, patient=np.array(pats),
             **{k: out[k] for k in KEYS})
    assert run_golden(name + ".pt", gp)
    print("%-16s members=%2d  ->  %s  (%d beats from %s)"
          % (name, len(ens), gp, len(pats), sorted(set(pats))))
    print("   duration (ms):", np.round(out["duration"], 1))
    print("   tau_on   (ms):", np.round(out["tau_on"], 1))
