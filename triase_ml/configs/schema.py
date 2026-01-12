from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

TaskType = Literal["diagnosis", "handling_no_diag", "handling_with_diag", "pipeline_diag_then_handling"]
CVType = Literal["single_split", "stratified_kfold"]
AggregationType = Literal["median", "mean"]

FeatureMethod = Literal[
    "none",
    "xgb_gain",
    "rf_importance",
    "lgbm_importance",
    "lasso_coef",
    "mutual_info",
    "chi2",
    "rfe_rf",
    "sfs_rf",
]

ModelName = Literal[
    "logreg",
    "rf",
    "knn",
    "gb",
    "hgb",
    "extra_trees",
    "adaboost",
    "bagging",
    "xgb",
    "lgbm",
    "mlp",
    "svm_linear",
    "svm_rbf",
    "lda",
    "voting",
    "stacking",
]

MetricName = Literal["macro_f1", "macro_recall", "macro_precision", "accuracy", "roc_auc_ovr"]


@dataclass
class DataConfig:
    excel_path: str
    sheet_name: str = "iccu_cleaned_imputed"
    random_seed: int = 42


@dataclass
class PreprocessConfig:
    include_diagnosis_feature: bool = False  # for handling_with_diag
    feature_method: FeatureMethod = "none"
    top_n_features: int = 20

    elimination_enabled: bool = False
    elimination_k: float = 0.7  # fraction per class to keep on training fold
    elimination_agg: AggregationType = "median"


@dataclass
class CVConfig:
    cv_type: CVType = "stratified_kfold"
    n_splits: int = 10
    test_size: float = 0.2  # used only for single_split
    shuffle: bool = True
    random_seed: int = 42


@dataclass
class TuningConfig:
    enabled: bool = False
    metric: MetricName = "macro_f1"

    # elimination k search
    k_grid: Sequence[float] = field(default_factory=lambda: (0.3, 0.5, 0.7, 0.9))

    # model hyperparameter grid (per model name)
    # Example: {"rf": {"n_estimators": [200, 500], "max_depth": [None, 10]}}
    model_param_grid: Dict[str, Dict[str, Sequence[object]]] = field(default_factory=dict)


@dataclass
class OutputConfig:
    out_dir: str = "outputs"
    save_models: bool = True
    save_predictions: bool = True


@dataclass
class ExperimentConfig:
    data: DataConfig
    preprocess: PreprocessConfig
    cv: CVConfig
    tuning: TuningConfig
    output: OutputConfig

    task: TaskType = "diagnosis"
    models: List[ModelName] = field(default_factory=lambda: ["rf", "xgb", "lgbm"])
    primary_model_for_explanations: Optional[ModelName] = None  # if None, use first in models
