from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..configs.schema import ExperimentConfig
from ..data.io import load_triase_excel
from ..features.selectors import select_features
from ..models.registry import build_models
from ..viz.plots import save_confusion_matrix, save_feature_importance_bar, save_roc_auc
from ..viz.shap_viz import save_shap_beeswarm
from .elimination import eliminate_by_centroid_distance
from .metrics import aggregate_fold_results, compute_metrics
from .tuning import tune_model_and_elimination


def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _encode_target(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder, List[str]]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    class_names = [str(c) for c in le.classes_]
    return y_enc, le, class_names


def _get_task_data(cfg: ExperimentConfig):
    data = load_triase_excel(cfg.data.excel_path, sheet_name=cfg.data.sheet_name)
    task = cfg.task

    if task == "diagnosis":
        X = data.X_no_diagnosis
        y = data.y_diagnosis
        return X, y, None, None

    if task == "handling_no_diag":
        X = data.X_no_diagnosis
        y = data.y_handling
        return X, y, None, None

    if task == "handling_with_diag":
        X = data.X_with_diagnosis
        y = data.y_handling
        return X, y, None, None

    if task == "pipeline_diag_then_handling":
        # stage1: diagnosis using no_diag
        # stage2: handling uses with_diagnosis (we overwrite diagnosis one-hot columns using predicted stage1)
        return data.X_no_diagnosis, data.y_diagnosis, data.X_with_diagnosis, data.y_handling

    raise ValueError(f"Unknown task: {task}")


def _fit_predict_model(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None
    return y_pred, y_proba


def run_experiment(cfg: ExperimentConfig) -> Dict[str, object]:
    out_dir = cfg.output.out_dir
    _ensure_dir(out_dir)

    # save config
    with open(Path(out_dir) / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    X, y, X_stage2, y_stage2 = _get_task_data(cfg)
    y_enc, le, class_names = _encode_target(y)

    models = build_models(cfg.models, seed=cfg.data.random_seed)
    primary_model_name = cfg.primary_model_for_explanations or cfg.models[0]

    results_all: Dict[str, object] = {"task": cfg.task, "models": {}}

    # ----------------------------
    # CV splits
    # ----------------------------
    if cfg.cv.cv_type == "single_split":
        idx = np.arange(len(y_enc))
        train_idx, test_idx = train_test_split(
            idx,
            test_size=cfg.cv.test_size,
            random_state=cfg.cv.random_seed,
            stratify=y_enc,
        )
        splits = [(train_idx, test_idx)]
    else:
        skf = StratifiedKFold(
            n_splits=cfg.cv.n_splits,
            shuffle=cfg.cv.shuffle,
            random_state=cfg.cv.random_seed,
        )
        splits = list(skf.split(X, y_enc))

    # ----------------------------
    # Train per model
    # ----------------------------
    for model_name, model in models.items():
        # Optional joint tuning of elimination k and model hyperparameters
        tuned_k = cfg.preprocess.elimination_k
        tuned_params = {}

        if cfg.tuning.enabled:
            grid = cfg.tuning.model_param_grid.get(model_name, {})
            tune_res = tune_model_and_elimination(
                base_model=model,
                X=X,
                y_enc=y_enc,
                splits=splits,
                feature_method=cfg.preprocess.feature_method,
                top_n_features=cfg.preprocess.top_n_features,
                elimination_enabled=cfg.preprocess.elimination_enabled,
                elimination_agg=cfg.preprocess.elimination_agg,
                k_grid=cfg.tuning.k_grid,
                model_param_grid=grid,
                metric=cfg.tuning.metric,
                random_seed=cfg.data.random_seed,
            )
            tuned_k = tune_res.get("best_k", tuned_k)
            tuned_params = tune_res.get("best_params", {}) or {}

            tune_path = Path(out_dir) / "tuning" / f"{model_name}.json"
            tune_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tune_path, "w", encoding="utf-8") as f:
                json.dump(tune_res, f, indent=2)

        if tuned_params:
            try:
                model.set_params(**tuned_params)
            except Exception:
                pass

        fold_results = {}
        y_true_all: List[int] = []
        y_pred_all: List[int] = []
        y_proba_all: List[np.ndarray] = []

        for fold_i, (tr_idx, te_idx) in enumerate(splits):
            X_tr = X.iloc[tr_idx].copy()
            X_te = X.iloc[te_idx].copy()
            y_tr = y_enc[tr_idx]
            y_te = y_enc[te_idx]

            # feature selection on training fold only
            fs = select_features(
                method=cfg.preprocess.feature_method,
                X_train=X_tr,
                y_train=pd.Series(y_tr),
                top_n=cfg.preprocess.top_n_features,
                random_seed=cfg.data.random_seed,
            )
            selected = fs.selected_features
            X_tr = X_tr[selected]
            X_te = X_te[selected]

            # optional elimination on training fold only
            if cfg.preprocess.elimination_enabled:
                X_tr_df = pd.DataFrame(X_tr, columns=selected)
                y_tr_s = pd.Series(y_tr)
                X_tr_df, y_tr_s = eliminate_by_centroid_distance(
                    X_tr_df, y_tr_s, k=tuned_k, agg=cfg.preprocess.elimination_agg
                )
                X_tr_np = X_tr_df.values
                y_tr_np = y_tr_s.values
                X_te_np = X_te.values
            else:
                X_tr_np = X_tr.values
                y_tr_np = y_tr
                X_te_np = X_te.values

            # scale (fits on train fold only)
            scaler = StandardScaler()
            X_tr_np = scaler.fit_transform(X_tr_np)
            X_te_np = scaler.transform(X_te_np)

            # task pipeline special-case
            if cfg.task == "pipeline_diag_then_handling":
                # stage 1: diagnosis model predicts diagnosis for train/test, then stage 2 predicts handling
                stage1_model = build_models([model_name], seed=cfg.data.random_seed)[model_name]
                stage1_model.fit(X_tr_np, y_tr_np)
                yhat_tr = stage1_model.predict(X_tr_np)
                yhat_te = stage1_model.predict(X_te_np)

                # stage2: build X2 from X_stage2 using indices tr/te and overwrite diag one-hots.
                X2_tr_full = X_stage2.iloc[tr_idx].copy()
                X2_te_full = X_stage2.iloc[te_idx].copy()

                diag_prefix = "Diagnosa Penyakit jantung pasien  (text) _"
                diag_cols = [c for c in X2_tr_full.columns if c.startswith(diag_prefix)]

                for df_part, yhat in [(X2_tr_full, yhat_tr), (X2_te_full, yhat_te)]:
                    if diag_cols:
                        df_part.loc[:, diag_cols] = 0
                        for i, cls in enumerate(le.classes_):
                            col = f"{diag_prefix}{cls}"
                            if col in df_part.columns:
                                df_part.loc[:, col] = (yhat == i).astype(int)

                # encode handling (stage2 target) consistently on the whole fold union
                y2_enc_all, le2, class_names2 = _encode_target(
                    pd.concat([y_stage2.iloc[tr_idx], y_stage2.iloc[te_idx]], axis=0)
                )
                y2_tr = le2.transform(y_stage2.iloc[tr_idx].astype(str))
                y2_te = le2.transform(y_stage2.iloc[te_idx].astype(str))

                fs2 = select_features(
                    method=cfg.preprocess.feature_method,
                    X_train=X2_tr_full,
                    y_train=pd.Series(y2_tr),
                    top_n=cfg.preprocess.top_n_features,
                    random_seed=cfg.data.random_seed,
                )
                sel2 = fs2.selected_features
                X2_tr = X2_tr_full[sel2]
                X2_te = X2_te_full[sel2]

                if cfg.preprocess.elimination_enabled:
                    X2_tr_df, y2_tr_s = eliminate_by_centroid_distance(
                        X2_tr, pd.Series(y2_tr), k=tuned_k, agg=cfg.preprocess.elimination_agg
                    )
                    X2_tr_np = X2_tr_df.values
                    y2_tr_np = y2_tr_s.values
                    X2_te_np = X2_te.values
                else:
                    X2_tr_np = X2_tr.values
                    y2_tr_np = y2_tr
                    X2_te_np = X2_te.values

                scaler2 = StandardScaler()
                X2_tr_np = scaler2.fit_transform(X2_tr_np)
                X2_te_np = scaler2.transform(X2_te_np)

                stage2_model = build_models([model_name], seed=cfg.data.random_seed)[model_name]
                y_pred, y_proba = _fit_predict_model(stage2_model, X2_tr_np, y2_tr_np, X2_te_np)

                fr = compute_metrics(y2_te, y_pred, y_proba=y_proba)
                fold_results[fold_i] = fr

                y_true_all.extend(list(y2_te))
                y_pred_all.extend(list(y_pred))
                if y_proba is not None:
                    y_proba_all.append(y_proba)

                class_names_plot = class_names2

                if cfg.output.save_models and fold_i == 0:
                    dump(
                        {"stage1": stage1_model, "scaler1": scaler, "stage2": stage2_model, "scaler2": scaler2},
                        Path(out_dir) / f"model_{model_name}_pipeline.joblib",
                        )
            else:
                y_pred, y_proba = _fit_predict_model(model, X_tr_np, y_tr_np, X_te_np)

                fr = compute_metrics(y_te, y_pred, y_proba=y_proba)
                fold_results[fold_i] = fr

                y_true_all.extend(list(y_te))
                y_pred_all.extend(list(y_pred))
                if y_proba is not None:
                    y_proba_all.append(y_proba)

                class_names_plot = class_names

                if cfg.output.save_models and fold_i == 0:
                    dump(
                        {"model": model, "scaler": scaler, "selected_features": selected},
                        Path(out_dir) / f"model_{model_name}.joblib",
                        )

        agg = aggregate_fold_results(fold_results)
        results_all["models"][model_name] = agg

        # save aggregated confusion matrix
        cm_path = Path(out_dir) / "figures" / model_name / "confusion_matrix.png"
        save_confusion_matrix(
            cm=agg["confusion_sum"],
            labels=class_names_plot,
            out_path=str(cm_path),
            title=f"{cfg.task} - {model_name} - Confusion (sum over folds)",
        )

        # save ROC curve if we have probas
        if len(y_proba_all) > 0:
            proba_concat = np.concatenate(y_proba_all, axis=0)
            roc_path = Path(out_dir) / "figures" / model_name / "roc_auc.png"
            save_roc_auc(
                y_true=np.asarray(y_true_all),
                y_proba=proba_concat,
                class_names=class_names_plot,
                out_path=str(roc_path),
                title=f"{cfg.task} - {model_name} - ROC (OvR)",
            )

        # save predictions
        if cfg.output.save_predictions:
            pred_path = Path(out_dir) / "predictions" / f"{model_name}.csv"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            dfp = pd.DataFrame({"y_true": y_true_all, "y_pred": y_pred_all})
            dfp.to_csv(pred_path, index=False)

    # ----------------------------
    # Feature importance figure (one split)
    # ----------------------------
    try:
        tr_idx, te_idx = splits[0]
        X_tr0 = X.iloc[tr_idx].copy()
        y_tr0 = y_enc[tr_idx]
        fs0 = select_features(
            method=cfg.preprocess.feature_method,
            X_train=X_tr0,
            y_train=pd.Series(y_tr0),
            top_n=cfg.preprocess.top_n_features,
            random_seed=cfg.data.random_seed,
        )
        if fs0.importance is not None:
            out_imp = Path(out_dir) / "figures" / "feature_importance.png"
            save_feature_importance_bar(
                importance=fs0.importance,
                out_path=str(out_imp),
                top_n=min(cfg.preprocess.top_n_features, 30),
                title=f"{cfg.task} - feature importance ({cfg.preprocess.feature_method})",
            )
    except Exception:
        pass

    # ----------------------------
    # SHAP explanations for primary model on first split
    # ----------------------------
    try:
        model_name = primary_model_name
        model = build_models([model_name], seed=cfg.data.random_seed)[model_name]

        tr_idx, te_idx = splits[0]
        X_tr = X.iloc[tr_idx].copy()
        X_te = X.iloc[te_idx].copy()
        y_tr = y_enc[tr_idx]

        fs = select_features(
            method=cfg.preprocess.feature_method,
            X_train=X_tr,
            y_train=pd.Series(y_tr),
            top_n=cfg.preprocess.top_n_features,
            random_seed=cfg.data.random_seed,
        )
        sel = fs.selected_features
        X_tr = X_tr[sel]
        X_te = X_te[sel]

        if cfg.preprocess.elimination_enabled:
            X_tr_elim, y_tr_s = eliminate_by_centroid_distance(
                X_tr, pd.Series(y_tr), k=cfg.preprocess.elimination_k, agg=cfg.preprocess.elimination_agg
            )
            X_tr = X_tr_elim
            y_tr = y_tr_s.values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr.values)
        X_te_s = scaler.transform(X_te.values)

        model.fit(X_tr_s, y_tr)

        n_explain = min(200, X_te.shape[0])
        X_explain = pd.DataFrame(X_te_s[:n_explain], columns=sel)
        X_bg = pd.DataFrame(X_tr_s, columns=sel)

        shap_path = Path(out_dir) / "figures" / model_name / "shap_beeswarm.png"
        save_shap_beeswarm(
            model=model,
            X_background=X_bg,
            X_explain=X_explain,
            out_path=str(shap_path),
            agg=cfg.preprocess.elimination_agg,
            max_display=min(30, len(sel)),
        )
    except Exception:
        pass

    with open(Path(out_dir) / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2, default=lambda o: None)

    return results_all
