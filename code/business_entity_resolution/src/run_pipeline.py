"""End-to-end entrypoint: normalise -> block -> train (features, XGBoost, thresholds) -> predict.

    python run_pipeline.py --stage all
    python run_pipeline.py --stage normalize|block|train|predict
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import config as C

# Order that reproduces the final submission (each step is its own process, so GPU and RAM
# are released between steps). v6 predictions provide the confident test pseudo-labels that
# v7's target encodings use; the final file is v7 with threshold t = 0.9.
STAGES = {
    "normalize": [["normalize.py", "--split", "train", "test"]],
    "block": [["blocking.py", "--split", "train", "test"]],
    "train": [
        ["ce.py", "--train"],                                   # MiniLM cross-encoder
        ["train.py", "--stage1-only"],                          # v1 + out-of-fold p1 (cached)
        ["ce.py", "--score", "train"],
        ["expand.py", "--n-s1", "500000"],                      # extra out-of-sample S1s for v2
        ["ce.py", "--score", "ext"],
        ["train.py", "--reuse", "--expand"],                    # v6 stage 2
    ],
    "predict": [
        ["predict.py", "--stage1-only"],                        # test v1 probabilities + spilled features
        ["ce.py", "--score", "test"],
        ["predict.py", "--reuse-v1", "--suffix", "_v6"],        # v6 test probabilities
        ["train.py", "--reuse", "--expand", "--pseudo-te"],     # v7 stage 2 (pseudo-label encodings)
        ["predict.py", "--reuse-v1", "--suffix", "_v7"],
        ["predict.py", "--write-only", "--strict", "0.9"],      # final matching_results.tsv + candidate_pairs.tsv
    ],
}


def main():
    """Run the requested stage(s) as separate processes (keeps peak memory per stage low)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    a = ap.parse_args()
    order = list(STAGES) if a.stage == "all" else [a.stage]
    for st in order:
        for cmd in STAGES[st]:
            print(f"=== {st}: {' '.join(cmd)}", flush=True)
            subprocess.run([sys.executable, *cmd], cwd=C.SRC_DIR, check=True)


if __name__ == "__main__":
    main()
