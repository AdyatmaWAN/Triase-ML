from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_selection import chi2, mutual_info_classif, RFE, SequentialFeatureSelector
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None  # type: ignore


@dataclass
class FeatureSelectionResult:
    selected_features: List[str]
    importance: Optional[pd.Series] = None  # index=feature, value=importance (if available)


def select_features(
    method: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 20,
    random_seed: int = 42,
) -> FeatureSelectionResult:
    method = (method or "none").lower()
    if method == "none":
        return FeatureSelectionResult(selected_features=list(X_train.columns), importance=None)

    if method == "mutual_info":
        mi = mutual_info_classif(X_train, y_train.values.ravel(), random_state=random_seed)
        s = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
        return FeatureSelectionResult(selected_features=s.head(top_n).index.tolist(), importance=s)

    if method == "chi2":
        # chi2 requires non-negative features. Data here is mostly 0/1 dummies + numeric (>=0 assumed).
        scores, _ = chi2(X_train, y_train)
        s = pd.Series(scores, index=X_train.columns).sort_values(ascending=False)
        return FeatureSelectionResult(selected_features=s.head(top_n).index.tolist(), importance=s)

    if method == "rf_importance":
        rf = RandomForestClassifier(n_estimators=300, random_state=random_seed, n_jobs=-1)
        rf.fit(X_train, y_train)
        s = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        return FeatureSelectionResult(selected_features=s.head(top_n).index.tolist(), importance=s)

    if method == "xgb_gain":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed but method='xgb_gain' was requested.")
        xgb = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=random_seed,
            n_jobs=-1,
        )
        xgb.fit(X_train, y_train)
        booster = xgb.get_booster()
        scores = booster.get_score(importance_type="gain")
        # XGB uses f0,f1,... by default if feature names not set. Ensure names set via DataFrame.
        # When using DataFrame, XGB should preserve feature names, but guard anyway:
        if any(k.startswith("f") and k[1:].isdigit() for k in scores.keys()):
            # map f{idx} to column name
            mapped = {}
            cols = list(X_train.columns)
            for k, v in scores.items():
                idx = int(k[1:])
                if 0 <= idx < len(cols):
                    mapped[cols[idx]] = v
            scores = mapped
        s = pd.Series(scores).sort_values(ascending=False)
        return FeatureSelectionResult(selected_features=s.head(top_n).index.tolist(), importance=s)

    if method == "lgbm_importance":
        if lgb is None:
            raise ImportError("lightgbm is not installed but method='lgbm_importance' was requested.")
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            random_state=random_seed,
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        s = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        return FeatureSelectionResult(selected_features=s.head(top_n).index.tolist(), importance=s)

    if method == "lasso_coef":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train.values)
        # For multiclass, Lasso is not a classifier. We use a regression proxy on encoded classes.
        # This is an approximation; interpret coefficients as heuristic.
        y_num = pd.Series(y_train).astype("category").cat.codes.values
        lasso = Lasso(alpha=0.05, random_state=random_seed, max_iter=5000)
        lasso.fit(Xs, y_num)
        s = pd.Series(np.abs(lasso.coef_), index=X_train.columns).sort_values(ascending=False)
        return FeatureSelectionResult(selected_features=s.head(top_n).index.tolist(), importance=s)

    if method == "rfe_rf":
        est = RandomForestClassifier(n_estimators=300, random_state=random_seed, n_jobs=-1)
        rfe = RFE(estimator=est, n_features_to_select=top_n)
        rfe.fit(X_train, y_train)
        feats = X_train.columns[rfe.support_].tolist()
        # importance not well-defined; return None
        return FeatureSelectionResult(selected_features=feats, importance=None)

    if method == "sfs_rf":
        est = RandomForestClassifier(n_estimators=300, random_state=random_seed, n_jobs=-1)
        sfs = SequentialFeatureSelector(estimator=est, n_features_to_select=top_n, direction="forward", n_jobs=-1)
        sfs.fit(X_train, y_train)
        feats = X_train.columns[sfs.get_support()].tolist()
        return FeatureSelectionResult(selected_features=feats, importance=None)

    raise ValueError(f"Unknown feature selection method: {method}")
