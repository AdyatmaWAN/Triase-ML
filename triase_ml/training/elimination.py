from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import pandas as pd


Aggregation = Literal["median", "mean"]


def eliminate_by_centroid_distance(
    X: pd.DataFrame,
    y: pd.Series,
    k: float,
    agg: Aggregation = "median",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Down-sample each class to keep int(len(class)*k) samples.

    Strategy: compute per-class centroid (median/mean) in feature space; for each sample in a class,
    compute its distance to the *other* class centroids and keep the closest ones (lowest summed distance).

    Notes:
    - This behaves like "prototype proximity" sampling. It is *not* SMOTE/ENN etc.
    - Apply ONLY on training folds to avoid leakage.
    """
    if not (0 < k <= 1.0):
        raise ValueError(f"k must be in (0,1], got {k}")

    X_arr = np.asarray(X.values, dtype=float)
    y_arr = np.asarray(y.values)

    classes = np.unique(y_arr)
    num_class = len(classes)
    if num_class < 2:
        return X.copy(), y.copy()

    # index samples per class
    X_div = []
    y_div = []
    for c in classes:
        idx = np.where(y_arr == c)[0]
        X_div.append(X_arr[idx])
        y_div.append(y_arr[idx])

    # compute centroids
    centroids = []
    for Xi in X_div:
        if agg == "median":
            cent = np.median(Xi, axis=0)
        elif agg == "mean":
            cent = np.mean(Xi, axis=0)
        else:
            raise ValueError(f"Unknown agg: {agg}")
        centroids.append(cent)

    # compute keep counts
    keep_counts = [max(1, int(len(Xi) * k)) for Xi in X_div]

    chosen_X = []
    chosen_y = []
    for i in range(num_class):
        Xi = X_div[i]
        yi = y_div[i]
        # sum distances to all other class centroids
        dist_sum = np.zeros((Xi.shape[0],), dtype=float)
        for j in range(num_class):
            if j == i:
                continue
            dist_sum += np.linalg.norm(Xi - centroids[j], axis=1)
        # keep closest
        order = np.argsort(dist_sum)
        keep_idx = order[: keep_counts[i]]
        chosen_X.append(Xi[keep_idx])
        chosen_y.append(yi[keep_idx])

    X_out = np.concatenate(chosen_X, axis=0)
    y_out = np.concatenate(chosen_y, axis=0)

    # restore DataFrame/Series
    X_df = pd.DataFrame(X_out, columns=X.columns)
    y_s = pd.Series(y_out, name=y.name)

    return X_df, y_s
