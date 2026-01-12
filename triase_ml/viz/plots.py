from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


def save_feature_importance_bar(
    importance: pd.Series,
    out_path: str,
    top_n: int = 20,
    title: str = "Feature importance",
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    s = importance.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    plt.barh(list(reversed(s.index.tolist())), list(reversed(s.values.tolist())))
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[str],
    out_path: str,
    title: str = "Confusion Matrix",
    normalize: Optional[str] = None,  # None, 'true', 'pred', 'all'
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels))
    disp.plot(include_values=True, cmap=None, ax=plt.gca(), colorbar=True, values_format=None, xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Sequence[str],
    out_path: str,
    title: str = "ROC Curves (OvR)",
) -> None:
    """Save ROC curves. Works for binary and multiclass (OvR)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    classes = np.unique(y_true)

    plt.figure(figsize=(7, 6))

    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    else:
        y_bin = label_binarize(y_true, classes=classes)
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            cname = class_names[int(i)] if i < len(class_names) else str(c)
            plt.plot(fpr, tpr, label=f"{cname} AUC={roc_auc:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
