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
    Platt/Isotonic scaling via CalibratedClassifierCV. We keep probability=False on SVC—
    calibration handles probabilities. Use a smaller cv for the calibrator to keep runtime sane.
    """
    # Underlying SVC
    base_svm = SVC(probability=False, random_state=random_state)

    # Wrap with calibrator; IMPORTANT: use estimator=... (not base_estimator)
    calibrated = CalibratedClassifierCV(
        estimator=base_svm,
        cv=3,
        method="sigmoid",
        n_jobs=-1,  # optional, supported by CalibratedClassifierCV
    )

    pipe = deepcopy(prep_scaled)
    pipe.steps.append(("clf", calibrated))

    # IMPORTANT: grid must target `estimator`, not `base_estimator`
    grid = {
        # Calibrator hyperparams (optional to tune)
        "clf__method": ["sigmoid", "isotonic"],  # isotonic can perform better with enough data
        "clf__cv": [3, 5],

        # Hyperparams of the underlying SVC live under `clf__estimator__*`
        "clf__estimator__kernel": ["rbf", "linear"],
        "clf__estimator__C": [0.5, 1, 4],
        "clf__estimator__gamma": ["scale", 0.01],  # ignored when kernel='linear', harmless in grid
        "clf__estimator__class_weight": [None, "balanced"],
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
