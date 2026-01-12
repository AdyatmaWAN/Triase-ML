from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


@dataclass
class FoldResult:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    roc_auc_ovr: Optional[float]
    confusion: np.ndarray


def compute_metrics(y_true, y_pred, y_proba=None) -> FoldResult:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    mp = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    mr = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    auc = None
    if y_proba is not None:
        try:
            classes = np.unique(y_true)
            if len(classes) == 2:
                # binary: roc_auc expects prob for positive class
                auc = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                # multiclass: OVR macro
                y_bin = label_binarize(y_true, classes=classes)
                auc = float(roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr"))
        except Exception:
            auc = None

    return FoldResult(
        accuracy=acc,
        macro_precision=mp,
        macro_recall=mr,
        macro_f1=mf1,
        roc_auc_ovr=auc,
        confusion=cm,
    )


def aggregate_fold_results(results: Dict[int, FoldResult]) -> Dict[str, object]:
    acc = np.mean([r.accuracy for r in results.values()])
    mp = np.mean([r.macro_precision for r in results.values()])
    mr = np.mean([r.macro_recall for r in results.values()])
    mf1 = np.mean([r.macro_f1 for r in results.values()])
    aucs = [r.roc_auc_ovr for r in results.values() if r.roc_auc_ovr is not None]
    auc = float(np.mean(aucs)) if len(aucs) else None
    cm = np.sum([r.confusion for r in results.values()], axis=0)

    return {
        "accuracy_mean": float(acc),
        "macro_precision_mean": float(mp),
        "macro_recall_mean": float(mr),
        "macro_f1_mean": float(mf1),
        "roc_auc_ovr_mean": auc,
        "confusion_sum": cm,
        "n_folds": len(results),
    }
