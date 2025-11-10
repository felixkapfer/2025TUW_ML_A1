# helpers.py

import os
import json
import random
import logging

import numpy as np
import pandas as pd

from dataclasses import dataclass

from typing import Dict, Any, Tuple, Optional


from sklearn.model_selection import train_test_split, StratifiedKFold

# -----------------------------
# Reproducibility & lightweight logging
# -----------------------------
def set_global_seed(seed: int = 42) -> None:
    """
    Set global/random seeds for reproducibility across numpy, python, and (optionally) torch.
    """
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # Torch not installed; that's fine.
        pass


class RunLogger:
    """
    Minimal JSONL logger for experiments. Each call to .log(dict) appends one JSON line.
    Keeps the experiment trail reproducible and auditable.
    """
    def __init__(self, path: str = "runs/step1_log.jsonl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def log(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")







# -----------------------------
# Data loading
# -----------------------------
@dataclass
class DatasetSpec:
    """
    Generic dataset spec. 
    - X_cols=None means 'use all columns except target_col and id_col (if provided)'.
    """
    csv_path: str
    target_col: Optional[str]
    id_col: Optional[str] = None
    X_cols: Optional[list[str]] = None

def load_dataset(spec: DatasetSpec) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """
    Load a CSV and split into:
        X : DataFrame of features
        y : Series of labels/targets or None (for test sets)
        ids: Series of IDs or None
    Rules:
    - If target_col is None or missing in the file, y=None (test mode).
    - If X_cols is None, X = all columns minus {target_col, id_col} (those present).
    - If X_cols is provided, X = df[X_cols] (and we do not auto-drop target/id).
    """
    df = pd.read_csv(spec.csv_path)
    df.columns = df.columns.str.strip()

    # IDs (optional)
    ids = df[spec.id_col] if (spec.id_col and spec.id_col in df.columns) else None

    # Determine if labels are available
    has_target = bool(spec.target_col) and spec.target_col in df.columns

    # Build X
    if spec.X_cols is None:
        # Use everything except target/id that actually exist
        drop_cols = []
        if has_target:
            drop_cols.append(spec.target_col)  # type: ignore[arg-type]
        if spec.id_col and spec.id_col in df.columns:
            drop_cols.append(spec.id_col)
        X = df.drop(columns=drop_cols, errors="ignore")
    else:
        # Respect explicit feature list
        X = df[spec.X_cols]

    # Build y (None if no target)
    y = df[spec.target_col] if has_target else None  # type: ignore[index]

    return X, y, ids




# -----------------------------
# Splitting strategy
# -----------------------------
@dataclass
class SplitConfig:
    """
    Holdout + inner CV configuration.
    """
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    inner_cv_k: int = 5
    shuffle: bool = True


def make_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: SplitConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create a single reproducible stratified holdout split for final reporting.
    """
    strat = y if cfg.stratify else None
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=strat
    )
    return X_train, X_holdout, y_train, y_holdout


def make_inner_cv(cfg: SplitConfig) -> StratifiedKFold:
    """
    Inner cross-validation object (to be used later inside GridSearchCV or custom loops).
    """
    return StratifiedKFold(
        n_splits=cfg.inner_cv_k,
        shuffle=cfg.shuffle,
        random_state=cfg.random_state
    )