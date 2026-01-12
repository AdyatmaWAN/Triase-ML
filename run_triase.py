from __future__ import annotations

import argparse
import json
from pathlib import Path

from triase_ml.configs.schema import (
    DataConfig,
    PreprocessConfig,
    CVConfig,
    TuningConfig,
    OutputConfig,
    ExperimentConfig,
)
from triase_ml.training.runner import run_experiment

import numpy as np

def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Triase ML runner (modular).")

    # data
    p.add_argument("--excel", required=True, help="Path to Triase_cleaned.xlsx (or equivalent).")
    p.add_argument("--sheet", default="iccu_cleaned_imputed", help="Excel sheet name.")

    # task
    p.add_argument(
        "--task",
        default="diagnosis",
        choices=["diagnosis", "handling_no_diag", "handling_with_diag", "pipeline_diag_then_handling"],
        help="Which prediction task to run.",
    )

    # models
    p.add_argument(
        "--models",
        default="rf,xgb,lgbm",
        help="Comma-separated model names. Example: rf,logreg,svm_rbf",
    )
    p.add_argument(
        "--primary_model",
        default=None,
        help="Model to use for SHAP (defaults to first model in --models).",
    )

    # feature selection
    p.add_argument(
        "--feature_method",
        default="none",
        choices=["none", "xgb_gain", "rf_importance", "lgbm_importance", "lasso_coef",
                 "mutual_info", "chi2", "rfe_rf", "sfs_rf"],
        help="Feature selection / importance method.",
    )
    p.add_argument("--top_n", type=int, default=20, help="Top-N features to keep (when applicable).")

    # elimination
    p.add_argument("--eliminate", action="store_true", help="Enable elimination (downsampling) on training folds.")
    p.add_argument("--k", type=float, default=0.7, help="Elimination fraction per class to keep (0<k<=1).")
    p.add_argument("--agg", default="median", choices=["median", "mean"], help="Centroid aggregation for elimination.")

    # CV
    p.add_argument("--cv", default="stratified_kfold", choices=["single_split", "stratified_kfold"])
    p.add_argument("--n_splits", type=int, default=10)
    p.add_argument("--test_size", type=float, default=0.2)

    # tuning
    p.add_argument("--tune", action="store_true", help="Enable tuning over elimination k and model hyperparameters.")
    p.add_argument("--k_grid", default="0.3,0.5,0.7,0.9", help="Comma-separated k values to try.")
    p.add_argument("--tune_metric", default="macro_f1", choices=["macro_f1", "macro_recall", "macro_precision", "accuracy", "roc_auc_ovr"])
    p.add_argument(
        "--param_grid_json",
        default=None,
        help=(
            "Optional JSON file containing per-model param grids. "
            'Format: {"rf": {"n_estimators": [200,500]}, "svm_rbf": {"C": [1,10]}}'
        ),
    )

    # output
    p.add_argument("--out", default="outputs", help="Output directory.")
    p.add_argument("--no_save_models", action="store_true", help="Do not save trained models.")
    p.add_argument("--no_save_predictions", action="store_true", help="Do not save per-sample predictions.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    k_grid = [float(x) for x in args.k_grid.split(",") if x.strip()]

    param_grid = {}
    if args.param_grid_json:
        with open(args.param_grid_json, "r", encoding="utf-8") as f:
            param_grid = json.load(f)

    cfg = ExperimentConfig(
        data=DataConfig(excel_path=args.excel, sheet_name=args.sheet),
        preprocess=PreprocessConfig(
            feature_method=args.feature_method,
            top_n_features=args.top_n,
            elimination_enabled=args.eliminate,
            elimination_k=args.k,
            elimination_agg=args.agg,
        ),
        cv=CVConfig(
            cv_type=args.cv,
            n_splits=args.n_splits,
            test_size=args.test_size,
        ),
        tuning=TuningConfig(
            enabled=args.tune,
            metric=args.tune_metric,
            k_grid=tuple(k_grid),
            model_param_grid=param_grid,
        ),
        output=OutputConfig(
            out_dir=args.out,
            save_models=not args.no_save_models,
            save_predictions=not args.no_save_predictions,
        ),
        task=args.task,
        models=model_list,
        primary_model_for_explanations=args.primary_model,
    )

    res = run_experiment(cfg)
    print(json.dumps(res, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
