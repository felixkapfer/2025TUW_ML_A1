# step5_ablation.py

from __future__ import annotations
from typing import Dict, Tuple, List, Any
from copy import deepcopy
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from .models import run_gridsearch

def add_rf_scaled_space(prep_scaled, random_state=42):
    rf_scaled = deepcopy(prep_scaled)
    rf_scaled.steps.append(("clf", RandomForestClassifier(random_state=random_state)))
    grid = {
        "clf__n_estimators": [200],
        "clf__max_depth": [None, 8],
        "clf__min_samples_leaf": [1, 2],
        "clf__class_weight": [None, "balanced_subsample"],
    }
    return ("RandomForest+Scaled", rf_scaled, grid)

def add_svm_calibrated_space(prep_scaled, random_state=42):
    """
    Platt scaling via CalibratedClassifierCV. Note: use smaller CV for the calibrator
    to keep runtime reasonable.
    """
    base_svm = SVC(probability=False, random_state=random_state)  # probability off; calibration handles probs
    calibrated = CalibratedClassifierCV(base_svm, cv=3, method="sigmoid")
    from sklearn.pipeline import Pipeline
    pipe = deepcopy(prep_scaled)
    pipe.steps.append(("clf", calibrated))
    grid = {
        "clf__base_estimator__C": [0.5, 1, 4],
        "clf__base_estimator__gamma": ["scale", 0.01],
        # class weights via base estimator:
        "clf__base_estimator__class_weight": [None, "balanced"],
    }
    return ("SVM+Calibrated", pipe, grid)

def run_small_ablations(
    X_train, y_train, inner_cv, primary="roc_auc", n_jobs=-1, verbose=1,
    spaces: List[Tuple[str, Any, Dict[str, List[Any]]]] = None
) -> pd.DataFrame:
    """
    Run a list of (name, pipeline, grid) and stack CV results for quick comparison.
    """
    results = []
    for (name, pipe, grid) in spaces:
        gs, res = run_gridsearch(name, pipe, grid, X_train, y_train, inner_cv, primary, n_jobs, verbose)
        results.append(res)
        print(f"[{name}] Best {primary.upper()} (CV):", res.loc[0, f"mean_test_{primary}"], "| Params:", gs.best_params_)
    return pd.concat(results, ignore_index=True)
