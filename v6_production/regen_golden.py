"""Recompute each bundle's golden outputs from the inputs already stored in it.

The golden files carry their own `window`/`emb`/`patient` arrays, so a rebuilt
bundle can be re-fixtured without re-extracting beats on a GPU. The inputs are
byte-identical to the originals; only the recorded outputs change, which is
exactly what should change when the weights do.
"""
import os

import numpy as np
import torch

from qrs_ensemble import QRSEnsemble, run_golden

KEYS = ("onset", "offset", "duration", "tau_on", "tau_off")

for name in ("megamodel_loo32", "allseed64"):
    gp = "golden_%s.npz" % name
    if not os.path.exists(gp):
        print("%-16s no existing golden to take inputs from, skipped" % name)
        continue
    g = np.load(gp, allow_pickle=True)
    window, emb = g["window"], g["emb"]
    patient = g["patient"] if "patient" in g.files else np.array([])

    ens = QRSEnsemble.load(name + ".pt")
    out = ens.predict(torch.as_tensor(window).float(),
                      torch.as_tensor(emb).float())
    old = {k: g[k] for k in KEYS}
    np.savez(gp, window=window, emb=emb, patient=patient,
             **{k: out[k] for k in KEYS})
    assert run_golden(name + ".pt", gp)

    shift = np.abs(out["duration"] - old["duration"]).max()
    print("%-16s members=%2d  beats=%d  golden OK  "
          "max |duration| change vs previous fixture: %.2f ms"
          % (name, len(ens), len(window), shift))
    print("   duration:", np.round(out["duration"], 1))
