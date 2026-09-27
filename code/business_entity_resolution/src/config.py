"""Central configuration: paths, hyper-parameters, seeds and caps (single place)."""
from pathlib import Path
import os
import random

import numpy as np

# Worker processes are always spawned (Windows default). On Linux the default fork start
# method can deadlock with polars' thread pool in the parent.
import multiprocessing as _mp
try:
    _mp.set_start_method("spawn")
except RuntimeError:
    pass

# ---------------------------------------------------------------- paths
SRC_DIR = Path(__file__).resolve().parent
PKG_DIR = SRC_DIR.parent                      # code/business_entity_resolution
REPO_DIR = PKG_DIR.parent.parent              # repo root
DATA_DIR = Path(os.environ.get("BER_DATA_DIR", REPO_DIR / "student_resource" / "dataset"))
CACHE_DIR = PKG_DIR / "cache"
MODELS_DIR = PKG_DIR / "models"
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT_DIR", REPO_DIR / "output"))
for _d in (CACHE_DIR, MODELS_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SPLITS = ("train", "test")

# ---------------------------------------------------------------- compute
SEED = 42
# Heavy numeric work (kNN search, model training/inference) runs on the GPU; CPU stages
# (normalisation, hashing, string features, stage 2) use 22 of the 28 logical cores.
N_JOBS = int(os.environ.get("BER_N_JOBS", min(22, (os.cpu_count() or 4) - 4)))
N_JOBS_INFER = int(os.environ.get("BER_N_JOBS_INFER", min(22, (os.cpu_count() or 4) - 4)))  # test inference
DEVICE = "cuda"          # torch / xgboost device (falls back to CPU when CUDA is unavailable)
KNN_PAIR_BUDGET = 30_000_000   # max product entries per GPU top-k batch (keeps VRAM < 8 GB)

# ---------------------------------------------------------------- blocking
MAX_BLOCK = 50           # exact-key blocks bigger than this (per side) are dropped
TFIDF_MIN_SIM = 0.05     # similarities below this are ignored in kNN
CAND_CAP = 40            # final candidates kept per S1 (recall-vs-cap curve printed by diagnostics)

# ---------------------------------------------------------------- model (XGBoost on CUDA)
XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    device=DEVICE,
    tree_method="hist",
    grow_policy="lossguide",   # leaf-wise like LightGBM
    max_leaves=127,
    max_depth=0,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    learning_rate=0.05,
    max_bin=256,
    seed=SEED,
    nthread=N_JOBS,
)
XGB_ROUNDS = 2000
XGB_EARLY_STOP = 50
VALID_FRAC = 0.2          # fraction of train S1 entities held out for validation
TRAIN_MAX_S1 = 600_000    # cap on S1 entities (with all their candidates) used for fitting


def set_seeds(seed: int = SEED) -> None:
    """Seed python and numpy RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
