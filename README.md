# Triase ML (modular)

This is a modular refactor of the original notebook-style scripts into a small package + CLI runner.

## Key capabilities

- Task selection:
  - `diagnosis`
  - `handling_no_diag`
  - `handling_with_diag`
  - `pipeline_diag_then_handling` (2-stage: diagnosis -> handling)

- Feature selection / feature importance:
  - `none`, `xgb_gain`, `rf_importance`, `lgbm_importance`, `lasso_coef`, `mutual_info`, `chi2`, `rfe_rf`, `sfs_rf`

- Elimination / downsampling:
  - Enable/disable
  - `k` fraction per class
  - centroid aggregation: `median` or `mean`

- CV:
  - `stratified_kfold` or `single_split`

- Hyperparameter tuning:
  - Joint tuning of elimination `k` and model hyperparameters (per-model grid)

- Outputs:
  - aggregated confusion matrix per model
  - ROC curves (when `predict_proba` available)
  - feature importance plot (when available from selector)
  - SHAP beeswarm plot for the primary model

Transformer models are intentionally excluded.

## Install (example)

```bash
pip install -U pandas numpy scikit-learn matplotlib shap joblib openpyxl
pip install -U xgboost lightgbm   # optional, only if you use xgb/lgbm or xgb_gain/lgbm_importance
```

## Run examples

### Diagnosis with 10-fold stratified CV, mutual information top-20

```bash
python run_triase.py --excel Triase_cleaned.xlsx --task diagnosis \
  --models rf,logreg,svm_rbf \
  --feature_method mutual_info --top_n 20 \
  --cv stratified_kfold --n_splits 10 \
  --out outputs/diagnosis_mi20
```

### Handling (with diagnosis), elimination enabled, tune k + RF params

Create `param_grid.json`:

```json
{
  "rf": {
    "n_estimators": [300, 600],
    "max_depth": [null, 10, 20],
    "min_samples_leaf": [1, 3]
  }
}
```

Run:

```bash
python run_triase.py --excel Triase_cleaned.xlsx --task handling_with_diag \
  --models rf \
  --feature_method rf_importance --top_n 30 \
  --eliminate --agg median --tune --k_grid 0.3,0.5,0.7,0.9 \
  --param_grid_json param_grid.json \
  --out outputs/handling_with_diag_tuned
```

## Output structure

- `config.json`
- `results_summary.json`
- `tuning/<model>.json` (if enabled)
- `figures/<model>/confusion_matrix.png`
- `figures/<model>/roc_auc.png` (if available)
- `figures/<model>/shap_beeswarm.png` (best-effort)
- `figures/feature_importance.png` (if available)
- `predictions/<model>.csv` (optional)
- `model_<model>.joblib` (optional)
