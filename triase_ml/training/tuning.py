from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from .elimination import eliminate_by_centroid_distance
from .metrics import compute_metrics
from ..features.selectors import select_features


def _score_from_foldresult(fr, metric: str) -> float:
    metric = metric.lower()
    if metric == "macro_f1":
        return fr.macro_f1
    if metric == "macro_recall":
        return fr.macro_recall
    if metric == "macro_precision":
        return fr.macro_precision
    if metric == "accuracy":
        return fr.accuracy
    if metric == "roc_auc_ovr":
        return fr.roc_auc_ovr if fr.roc_auc_ovr is not None else float("-inf")
    raise ValueError(f"Unknown metric: {metric}")


def tune_model_and_elimination(
    base_model,
    X: pd.DataFrame,
    y_enc: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    feature_method: str,
    top_n_features: int,
    elimination_enabled: bool,
    elimination_agg: str,
    k_grid: Sequence[float],
    model_param_grid: Dict[str, Sequence[object]],
    metric: str = "macro_f1",
    random_seed: int = 42,
) -> Dict[str, object]:
    """Grid search over (k, model params) using CV splits.

    Important:
    - Feature selection is refit on each training fold to avoid leakage.
    - Elimination (downsampling) is applied only on training folds.
    - Scaling is fit only on training folds.

    Returns dict with best_params, best_score, and a small leaderboard.
    """
    if not elimination_enabled:
        k_grid = (1.0,)  # no elimination

    # Build cartesian product of model params
    keys = list(model_param_grid.keys())
    values = [list(model_param_grid[k]) for k in keys]
    if len(keys) == 0:
        param_sets = [dict()]
    else:
        param_sets = []
        def rec(i, cur):
            if i == len(keys):
                param_sets.append(cur.copy())
                return
            for v in values[i]:
                cur[keys[i]] = v
                rec(i + 1, cur)
        rec(0, {})

    leaderboard = []
    best = {"score": float("-inf"), "k": None, "params": None}

    for k in k_grid:
        for params in param_sets:
            scores = []
            for tr_idx, te_idx in splits:
                X_tr = X.iloc[tr_idx].copy()
                X_te = X.iloc[te_idx].copy()
                y_tr = y_enc[tr_idx]
                y_te = y_enc[te_idx]

                fs = select_features(feature_method, X_tr, pd.Series(y_tr), top_n_features, random_seed=random_seed)
                sel = fs.selected_features
                X_tr = X_tr[sel]
                X_te = X_te[sel]

                if elimination_enabled:
                    X_tr, y_tr_s = eliminate_by_centroid_distance(X_tr, pd.Series(y_tr), k=k, agg=elimination_agg)
                    y_tr = y_tr_s.values

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr.values)
                X_te_s = scaler.transform(X_te.values)

                m = clone(base_model)
                if params:
                    m.set_params(**params)
                m.fit(X_tr_s, y_tr)
                y_pred = m.predict(X_te_s)
                y_proba = None
                if hasattr(m, "predict_proba"):
                    try:
                        y_proba = m.predict_proba(X_te_s)
                    except Exception:
                        y_proba = None
                fr = compute_metrics(y_te, y_pred, y_proba=y_proba)
                scores.append(_score_from_foldresult(fr, metric))

            mean_score = float(np.mean(scores))
            leaderboard.append({"k": float(k), "params": params, "score_mean": mean_score})
            if mean_score > best["score"]:
                best = {"score": mean_score, "k": float(k), "params": params}

    leaderboard = sorted(leaderboard, key=lambda d: d["score_mean"], reverse=True)[:20]
    return {
        "best_score": best["score"],
        "best_k": best["k"],
        "best_params": best["params"] or {},
        "leaderboard_top20": leaderboard,
    }
