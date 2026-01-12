from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# SHAP can be heavy; import lazily
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pick_background(X: pd.DataFrame, agg: str = "median") -> np.ndarray:
    if agg == "mean":
        return X.mean(axis=0).values.reshape(1, -1)
    return X.median(axis=0).values.reshape(1, -1)


def save_shap_beeswarm(
    model,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    out_path: str,
    agg: str = "median",
    max_display: int = 20,
) -> None:
    """Create and save SHAP beeswarm plot.

    Uses:
    - TreeExplainer for tree-based models when possible,
    - LinearExplainer for linear models with coef_,
    - Fallback to shap.Explainer with a background vector (median/mean).
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Choose explainer
    explainer = None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
    except Exception:
        try:
            if hasattr(model, "coef_"):
                explainer = shap.LinearExplainer(model, X_background, feature_perturbation="interventional")
                shap_values = explainer.shap_values(X_explain)
            else:
                bg = _pick_background(X_background, agg=agg)
                f = lambda z: model.predict_proba(z)  # noqa: E731
                explainer = shap.Explainer(f, bg)
                shap_values = explainer(X_explain)
        except Exception:
            # last resort: KernelExplainer (slow). Sample small set.
            bg = _pick_background(X_background, agg=agg)
            f = lambda z: model.predict_proba(z)  # noqa: E731
            explainer = shap.KernelExplainer(f, bg)
            shap_values = explainer.shap_values(X_explain, nsamples=200)

    plt.figure(figsize=(10, 6))
    try:
        shap.summary_plot(shap_values, X_explain, show=False, max_display=max_display)
    except Exception:
        # SHAP may return list for multiclass; pick class 0 summary by default
        if isinstance(shap_values, list) and len(shap_values) > 0:
            shap.summary_plot(shap_values[0], X_explain, show=False, max_display=max_display)
        else:
            raise
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
