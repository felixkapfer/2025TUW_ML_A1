# step3_models.py

from __future__ import annotations
from typing import Dict, Tuple, List, Any, Optional
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    make_scorer, roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


# -----------------------------
# Scoring dictionary (reusable)
# -----------------------------
def get_scorers() -> Dict[str, Any]:
    """
    Return a unified scoring dict for GridSearchCV.
    """
    return {
        "accuracy":   make_scorer(accuracy_score),
        "precision":  make_scorer(precision_score),
        "recall":     make_scorer(recall_score),
        "f1":         make_scorer(f1_score),
        "roc_auc":    "roc_auc",  # sklearn built-in string scorer
    }


# -----------------------------
# Build model spaces (pipelines + param grids)
# -----------------------------
def build_model_spaces(
    prep_scaled: Pipeline,
    prep_unscaled: Pipeline,
    random_state: int = 42
) -> Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]]:
    """
    Create three model spaces (RandomForest, SVM, GradientBoosting), each as:
      name -> (pipeline_with_classifier, param_grid)

    We pass in two preprocessors so we can:
      - Use scaling for SVM/GB
      - Optionally compare RF with and without scaling (start with unscaled baseline)
    """
    # Random Forest (commonly fine without scaling)
    rf_pipe = deepcopy(prep_unscaled)
    rf_pipe.steps.append(("clf", RandomForestClassifier(random_state=random_state)))
    rf_grid = {
        "clf__n_estimators": [400, 800, 1200],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_leaf": [1, 2, 3],
        "clf__max_features": ["sqrt", "log2"],
        "clf__class_weight": [None, "balanced", "balanced_subsample"],
}

    # SVM (needs scaling; probability=True for ROC-AUC and calibrated outputs)
    svm_pipe = deepcopy(prep_scaled)
    svm_pipe.steps.append(("clf", SVC(probability=True, random_state=random_state)))
    svm_grid = {
        "clf__C": [0.5, 1, 4],
        "clf__gamma": ["scale", 0.1, 0.01],
        "clf__class_weight": [None, "balanced"],
        # kernel is fixed to "rbf" (default) in SVC; add if you want to explore more
    }

    # Gradient Boosting (tree-based but benefits from consistent scaling step in our pipeline)
    gb_pipe = deepcopy(prep_scaled)
    gb_pipe.steps.append(("clf", GradientBoostingClassifier(random_state=random_state)))
    gb_grid = {
        "clf__n_estimators": [100, 300],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [2, 3],
    }

    return {
        "RandomForest": (rf_pipe, rf_grid),
        "SVM":          (svm_pipe, svm_grid),
        "GradientBoost":(gb_pipe, gb_grid),
    }


# -----------------------------
# Generic grid search runner
# -----------------------------
def run_gridsearch(
    name: str,
    pipe: Pipeline,
    param_grid: Dict[str, List[Any]],
    X_train,
    y_train,
    cv,
    primary_metric: str = "roc_auc",
    n_jobs: int = -1,
    verbose: int = 1,
) -> Tuple[GridSearchCV, pd.DataFrame]:
    """
    Run GridSearchCV with our common scoring dict and return both the fitted search object
    and a tidy DataFrame of CV results.
    """
    scorers = get_scorers()
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scorers,
        refit=primary_metric,   # this decides which scorer guides best_estimator_
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    gs.fit(X_train, y_train)

    # Assemble a compact results table
    res = pd.DataFrame(gs.cv_results_)
    keep_cols = [
        "mean_test_accuracy", "std_test_accuracy",
        "mean_test_precision", "std_test_precision",
        "mean_test_recall", "std_test_recall",
        "mean_test_f1", "std_test_f1",
        "mean_test_roc_auc", "std_test_roc_auc",
        "mean_fit_time", "std_fit_time",
        "params"
    ]
    present_cols = [c for c in keep_cols if c in res.columns]
    res_tidy = res[present_cols].sort_values(
        f"mean_test_{primary_metric}", ascending=False
    ).reset_index(drop=True)

    # Add model name for tracking
    res_tidy.insert(0, "model", name)
    return gs, res_tidy


# -----------------------------
# Safe proba/decision helpers
# -----------------------------
def predict_proba_or_score(
    estimator: Pipeline,
    X
) -> np.ndarray:
    """
    Return probability for the positive class if available; 
    fall back to decision_function rescaled to [0,1] if needed.
    """
    # Best effort: prefer probabilities
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba is not None:
            # Assume binary classification; use column 1
            return proba[:, 1]

    # Fallback to decision_function -> min-max to [0,1]
    if hasattr(estimator, "decision_function"):
        d = estimator.decision_function(X).astype(float)
        d_min, d_max = d.min(), d.max()
        if d_max > d_min:
            return (d - d_min) / (d_max - d_min)
        # degenerate case -> all same score
        return np.full_like(d, 0.5, dtype=float)

    # If neither exists, fallback to hard predictions -> {0,1} cast to float
    preds = estimator.predict(X)
    return preds.astype(float)


# -----------------------------
# Holdout evaluation
# -----------------------------
def evaluate_holdout(
    estimator: Pipeline,
    X_holdout,
    y_holdout
) -> Dict[str, float]:
    """
    Compute Accuracy, Precision, Recall, F1, ROC-AUC on the holdout split.
    """
    y_pred = estimator.predict(X_holdout)
    y_proba = predict_proba_or_score(estimator, X_holdout)

    metrics = {
        "accuracy":  float(accuracy_score(y_holdout, y_pred)),
        "precision": float(precision_score(y_holdout, y_pred)),
        "recall":    float(recall_score(y_holdout, y_pred)),
        "f1":        float(f1_score(y_holdout, y_pred)),
        "roc_auc":   float(roc_auc_score(y_holdout, y_proba)),
    }
    return metrics


# -----------------------------
# Kaggle submission helper
# -----------------------------
def make_kaggle_submission(
    estimator: Pipeline,
    X_test,
    test_ids: Optional[pd.Series],
    out_path: str = "kaggle_submission.csv",
    positive_label_name: str = "class",
    output: str = "label",
    threshold: float = 0.5
) -> str:
    """
    Create a Kaggle CSV.

    Columns:
      - "ID"      (uppercase)
      - positive_label_name (e.g. "class")
    output:
      - "label": hard 0/1 labels
      - "proba": probabilities
    """
    # 1) Get scores/probabilities
    values = None
    try:
        # Binary: take the column for the positive class
        proba = estimator.predict_proba(X_test)
        # If class order isn't [0,1], [:,1] will still refer to the positive class
        values = proba[:, 1]
    except Exception:
        # Fallback for models without predict_proba
        try:
            scores = estimator.decision_function(X_test)
            # For "proba" you could normalize, but we leave them as scores
            values = scores
        except Exception:
            # Last fallback: direct prediction
            preds = estimator.predict(X_test)
            values = preds.astype(int)

    # 2) Optional: hard labels
    if output == "label":
        # If values are already 0/1 they remain; otherwise threshold
        if values.dtype.kind in "fc":  # float/complex -> threshold
            values = (values >= threshold).astype(int)
        else:
            values = values.astype(int)

    # 3) Set IDs (uppercase "ID")
    if test_ids is None:
        ids = pd.Series(np.arange(len(values)), name="ID")
    else:
        ids = test_ids.rename("ID")

    # 4) Build DataFrame (columns exactly "ID" and positive_label_name)
    sub = pd.DataFrame({"ID": ids.values, positive_label_name: values})
    if output == "label":
        sub[positive_label_name] = sub[positive_label_name].astype(int)

    sub.to_csv(out_path, index=False)
    return out_path
