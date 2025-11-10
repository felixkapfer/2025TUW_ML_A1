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
    target_col: str
    id_col: Optional[str] = None
    X_cols: Optional[list[str]] = None

def load_dataset(spec: DatasetSpec) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """
    Load a CSV, split into X (features), y (target), and optional id series.
    Returns:
        X: DataFrame of features
        y: Series of labels/targets
        ids: Series of IDs or None
    """
    df = pd.read_csv(spec.csv_path)
    ids = df[spec.id_col] if spec.id_col and spec.id_col in df.columns else None
    if spec.X_cols is None:
        drop_cols = [spec.target_col] + ([spec.id_col] if spec.id_col and spec.id_col in df.columns else [])
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    else:
        X = df[spec.X_cols]
    y = df[spec.target_col]
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