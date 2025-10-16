#!/usr/bin/env python
from __future__ import annotations
import argparse
from mlc.config import load_config
from mlc.trainer import run_training

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_training(cfg)

if __name__ == "__main__":
    main()
