# preprocessing.py

from __future__ import annotations
from typing import Optional, Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Optional SMOTE support (only if installed)
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    _IMBLEARN_AVAILABLE = True
except Exception:
    _IMBLEARN_AVAILABLE = False


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """
    No-op transformer for cases where we want a switchable step.
    """
    def fit(self, X, y=None): 
        return self
    def transform(self, X): 
        return X


def make_numeric_preprocessor(
    numeric_cols: Optional[list[str]] = None,
    impute_strategy: Literal["median", "mean", "most_frequent"] = "median",
    with_scaler: bool = True
) -> ColumnTransformer:
    """
    Build a ColumnTransformer for numeric preprocessing:
      - SimpleImputer (default median)
      - optional StandardScaler
    If numeric_cols is None, we assume all features are numeric and pass remainder='drop'.
    """
    steps = [
        ("impute", SimpleImputer(strategy=impute_strategy)),
        ("scale", StandardScaler() if with_scaler else IdentityTransformer())
    ]

    # If columns aren’t pre-specified, treat all as numeric
    transformer = Pipeline(steps)
    if numeric_cols is None:
        # When all features are numeric, we can just apply the pipeline directly.
        # ColumnTransformer here wraps "all columns" behavior.
        return ColumnTransformer(
            transformers=[("num", transformer, slice(0, None))],
            remainder="drop"
        )
    else:
        return ColumnTransformer(
            transformers=[("num", transformer, numeric_cols)],
            remainder="drop"
        )


def make_preprocess_pipeline(
    numeric_cols: Optional[list[str]] = None,
    impute_strategy: Literal["median","mean","most_frequent"] = "median",
    with_scaler: bool = True,
    use_smote: bool = False,
    smote_kwargs: Optional[dict] = None
):
    """
    Create a preprocessing pipeline for numeric-only problems.

    Args:
        numeric_cols: list of numeric column names; if None, assumes all columns are numeric.
        impute_strategy: how to impute missing numeric values.
        with_scaler: whether to add StandardScaler.
        use_smote: if True and imbalanced-learn is available, wrap the pipeline so SMOTE is applied in the training folds only.
        smote_kwargs: kwargs passed to SMOTE (e.g., sampling_strategy, k_neighbors). Ignored if use_smote=False.

    Returns:
        A sklearn/imbalanced-learn Pipeline (depending on use_smote) that produces preprocessed features.
        You will append your classifier as the last step in later steps.
    """
    numeric = make_numeric_preprocessor(
        numeric_cols=numeric_cols,
        impute_strategy=impute_strategy,
        with_scaler=with_scaler
    )

    if use_smote:
        if not _IMBLEARN_AVAILABLE:
            raise ImportError("imblearn is not installed. Install with `pip install imbalanced-learn`.")
        # SMOTE must be *before* the classifier and after imputation/scaling; 
        # Using ImbPipeline ensures resampling occurs only on the training folds within CV.
        return ImbPipeline(steps=[
            ("prep", numeric),
            ("smote", SMOTE(**(smote_kwargs or {})))
            # ("clf", ...) will be appended later
        ])
    else:
        return Pipeline(steps=[
            ("prep", numeric)
            # ("clf", ...) will be appended later
        ])
