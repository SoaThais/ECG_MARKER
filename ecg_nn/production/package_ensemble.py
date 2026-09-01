"""Pack trained heads from one or more run directories into a shipping bundle.

    python package_ensemble.py --out production/megamodel_loo32.pt \\
        --ckpt final.pt runs/v6_lin_nohub_s2at1000
    python package_ensemble.py --out production/allseed64.pt \\
        --ckpt ep00100.pt runs/v6_ship_all --all_patients

Every member must share one architecture, which is asserted rather than assumed:
mixing the mlp64/all heads (373,196 params) with the linear/no_hubert ones
(331,606) would load fine per-member and produce an ensemble whose members read
different feature sets.

provenance records, per member, which run and which patient it was held out
from. That field is what keeps an honest evaluation possible later: at inference
a fold member and an all-patient member are interchangeable weights, but on a
patient the latter was trained on they are not, and once the record is gone it
cannot be reconstructed.
"""
import argparse
import glob
import hashlib
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCHEMA = 1

# The architecture every shipped member must have. Kept explicit here rather
# than inferred from a checkpoint so that a wrong bundle fails loudly at build
# time instead of at the bedside.
MODEL_KWARGS = dict(
    window_size=None,          # filled from --window_size
    embed_dim=768,
    scale=1.0,
    scorer_scale=0.5,
    tau_cap=None,
    tau_floor=1.0,
    tau_c_init_on=15.0,
    tau_c_init_off=15.0,
    tau_detach=False,          # runs used --tau_no_detach
    offset_prior=False,        # runs used --no_offset_prior
    no_compress=False,
    compress_mode="full",
    detach_routing=False,
    center_context=False,
    use_ste=False,
    feature_pool=False,
    tau_arch="linear",
    tau_feats="no_hubert",
)


def fold_of(cell):
    m = re.search(r"holdout_patient(p\d+)", cell)
    if m:
        return m.group(1)
    return cell if re.fullmatch(r"p\d+", cell) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--window_size", type=int, default=0,
                    help="0 = infer from the checkpoint's t_grid buffer, which "
                         "is the only source that cannot disagree with the "
                         "weights being packaged")
    ap.add_argument("--all_patients", action="store_true",
                    help="members trained on every patient; held_out is recorded "
                         "as None so no future evaluation mistakes them for "
                         "clean on any patient")
    ap.add_argument("--encoder_sha256", default="",
                    help="hash of the HuBERT weights these members expect")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    kw = dict(MODEL_KWARGS)

    members, prov, sizes, wsizes = [], [], set(), set()
    for run in args.runs:
        pat = os.path.join(run, "cells", "*", "head*_seed*", args.ckpt)
        for p in sorted(glob.glob(pat)):
            head_dir = os.path.basename(os.path.dirname(p))
            cell = os.path.basename(os.path.dirname(os.path.dirname(p)))
            sd = torch.load(p, map_location="cpu", weights_only=False)
            n = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
            sizes.add(n)
            if "t_grid" in sd:
                wsizes.add(int(sd["t_grid"].shape[-1]))
            if args.fp16:
                sd = {k: (v.half() if v.is_floating_point() else v)
                      for k, v in sd.items()}
            members.append(sd)
            prov.append({
                "run": os.path.basename(run.rstrip("/")),
                "cell": cell,
                "held_out": None if args.all_patients else fold_of(cell),
                "seed": int(head_dir.split("_seed")[-1]),
                "ckpt": args.ckpt,
            })

    if not members:
        raise SystemExit("no %s found under %s" % (args.ckpt, args.runs))
    if len(sizes) != 1:
        raise SystemExit("members disagree on parameter count: %s -- refusing "
                         "to mix architectures" % sorted(sizes))
    if len(wsizes) > 1:
        raise SystemExit("members disagree on window_size: %s" % sorted(wsizes))
    kw["window_size"] = args.window_size or (sorted(wsizes)[0] if wsizes else 0)
    if not kw["window_size"]:
        raise SystemExit("could not determine window_size; pass --window_size")

    blob = {
        "schema": SCHEMA,
        "model_kwargs": kw,
        "members": members,
        "provenance": prov,
        "manifest": {
            "combine": "median",
            "n_members": len(members),
            "param_count": sorted(sizes)[0],
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_runs": [os.path.basename(r.rstrip("/")) for r in args.runs],
            "ckpt": args.ckpt,
            "dtype": "fp16" if args.fp16 else "fp32",
            "encoder_sha256": args.encoder_sha256,
            "all_patients": bool(args.all_patients),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(blob, args.out)
    held = sorted({str(p["held_out"]) for p in prov})
    print("wrote %s\n  members=%d  params/member=%d  held_out=%s  size=%.1f MB"
          % (args.out, len(members), sorted(sizes)[0], held,
             os.path.getsize(args.out) / 1e6))


if __name__ == "__main__":
    main()
