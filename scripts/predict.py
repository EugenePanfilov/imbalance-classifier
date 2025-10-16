#!/usr/bin/env python
from __future__ import annotations
import argparse
import pandas as pd
from mlc.config import load_config
from mlc.infer import InferenceModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (paths.artifacts_dir)")
    ap.add_argument("--input", required=True, help="CSV with features")
    ap.add_argument("--out", required=True, help="Where to save predictions CSV")
    args = ap.parse_args()

    cfg = load_config(args.config)
    inf = InferenceModel.load(cfg.paths)
    df = pd.read_csv(args.input)
    proba, label = inf.predict(df)
    out = df.copy()
    out[proba.name] = proba.values
    out[label.name] = label.values
    out.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
