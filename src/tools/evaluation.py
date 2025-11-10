# evaluation.py

from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix, RocCurveDisplay,
    PrecisionRecallDisplay, ConfusionMatrixDisplay, brier_score_loss
)
from sklearn.calibration import CalibrationDisplay

import matplotlib.pyplot as plt

def summarize_cv_table(cv_table: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    Return a compact top-k leaderboard across all models by ROC-AUC.
    """
    sort_col = "mean_test_roc_auc"
    cols = [
        "model",
        "mean_test_accuracy","std_test_accuracy",
        "mean_test_precision","std_test_precision",
        "mean_test_recall","std_test_recall",
        "mean_test_f1","std_test_f1",
        "mean_test_roc_auc","std_test_roc_auc",
        "mean_fit_time","std_fit_time",
        "params"
    ]
    present = [c for c in cols if c in cv_table.columns]
    out = cv_table.sort_values(sort_col, ascending=False)[present].head(top_k).reset_index(drop=True)
    return out


def summarize_holdout(holdout_scores: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Turn the holdout dict into a table (one row per model).
    """
    rows = []
    for name, scores in holdout_scores.items():
        row = {"model": name}
        row.update(scores)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return df


def threshold_sweep(y_true, y_scores, metric="f1", grid_size: int = 101) -> Tuple[float, float]:
    """
    Search threshold in [0,1] to maximize a chosen metric ('f1','precision','recall','accuracy').
    Returns (best_threshold, best_value).
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    thresholds = np.linspace(0, 1, grid_size)
    best_thr, best_val = 0.5, -np.inf
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        if metric == "f1":
            val = f1_score(y_true, y_pred)
        elif metric == "precision":
            val = precision_score(y_true, y_pred)
        elif metric == "recall":
            val = recall_score(y_true, y_pred)
        elif metric == "accuracy":
            val = accuracy_score(y_true, y_pred)
        else:
            raise ValueError("Unknown metric for sweep.")
        if val > best_val:
            best_thr, best_val = t, val
    return best_thr, best_val


def plot_curves(estimator, X_holdout, y_holdout, title_prefix: str = "") -> None:
    """
    Draw ROC, PR, Confusion (at 0.5), and Calibration plots for the holdout.
    """
    # proba/score
    if hasattr(estimator, "predict_proba"):
        y_scores = estimator.predict_proba(X_holdout)[:, 1]
    elif hasattr(estimator, "decision_function"):
        d = estimator.decision_function(X_holdout).astype(float)
        d_min, d_max = d.min(), d.max()
        y_scores = (d - d_min) / (d_max - d_min) if d_max > d_min else np.full_like(d, 0.5, dtype=float)
    else:
        y_scores = estimator.predict(X_holdout).astype(float)

    y_pred = (y_scores >= 0.5).astype(int)

    # ROC
    RocCurveDisplay.from_predictions(y_holdout, y_scores)
    plt.title(f"{title_prefix} ROC curve")
    plt.show()

    # PR
    PrecisionRecallDisplay.from_predictions(y_holdout, y_scores)
    plt.title(f"{title_prefix} Precision-Recall curve")
    plt.show()

    # Confusion @ 0.5
    ConfusionMatrixDisplay.from_predictions(y_holdout, y_pred)
    plt.title(f"{title_prefix} Confusion Matrix (thr=0.5)")
    plt.show()

    # Calibration
    try:
        prob_true, prob_pred = CalibrationDisplay.from_predictions(y_holdout, y_scores)
        plt.title(f"{title_prefix} Calibration (Reliability)")
        plt.show()
    except Exception:
        pass

    # Brier
    try:
        bs = brier_score_loss(y_holdout, y_scores)
        print(f"{title_prefix} Brier score: {bs:.4f}")
    except Exception:
        pass
