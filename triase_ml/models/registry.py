from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None  # type: ignore


@dataclass
class ModelSpec:
    name: str
    estimator: object


def build_base_models(seed: int = 42) -> Dict[str, object]:
    models: Dict[str, object] = {
        "logreg": LogisticRegression(max_iter=2000, random_state=seed, n_jobs=-1),
        "rf": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "gb": GradientBoostingClassifier(random_state=seed),
        "hgb": HistGradientBoostingClassifier(max_iter=500, random_state=seed),
        "extra_trees": ExtraTreesClassifier(n_estimators=500, random_state=seed, n_jobs=-1),
        "adaboost": AdaBoostClassifier(n_estimators=300, random_state=seed),
        "bagging": BaggingClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "mlp": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=seed),
        "svm_linear": SVC(kernel="linear", probability=True, random_state=seed),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=seed),
        "lda": LinearDiscriminantAnalysis(),
    }
    if XGBClassifier is not None:
        models["xgb"] = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        )
    if lgb is not None:
        models["lgbm"] = lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            random_state=seed,
            verbose=-1,
            n_jobs=-1,
        )
    return models


def build_ensemble_models(base_models: Dict[str, object], seed: int = 42) -> Dict[str, object]:
    # For ensembles, use a subset of base models that have stable predict_proba
    est_list = [(n, m) for n, m in base_models.items() if n in {"logreg", "rf", "xgb", "lgbm", "svm_rbf", "lda"}]
    ensembles: Dict[str, object] = {}
    if len(est_list) >= 2:
        ensembles["voting"] = VotingClassifier(estimators=est_list, voting="hard", n_jobs=-1)
        ensembles["stacking"] = StackingClassifier(
            estimators=est_list,
            final_estimator=LogisticRegression(max_iter=2000, random_state=seed, n_jobs=-1),
            n_jobs=-1,
        )
    return ensembles


def build_models(model_names: List[str], seed: int = 42) -> Dict[str, object]:
    base = build_base_models(seed=seed)
    ens = build_ensemble_models(base, seed=seed)
    all_models = {**base, **ens}
    missing = [m for m in model_names if m not in all_models]
    if missing:
        raise ValueError(f"Unknown or unavailable models: {missing}. Available={sorted(all_models.keys())}")
    return {m: all_models[m] for m in model_names}
