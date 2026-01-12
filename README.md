# Triase ML (modular)

A small, modular Python package plus a CLI runner for running supervised classification experiments on the **Triase_cleaned.xlsx** dataset.

## Core Features
- **Tasks**: Diagnosis and Handling (including a 2-stage pipeline).
- **Feature Selection**: Fit on training folds only (supports various methods).
- **Elimination**: Optional class-wise downsampling on training folds.
- **Cross-Validation**: `single_split` or `stratified_kfold`.
- **Tuning**: Joint tuning of elimination ratio (`k`) and model hyperparameters.
- **Outputs**: Metrics summary, confusion matrices, ROC curves, feature importance, and SHAP global explanations.

## Repository Layout
- `run_triase.py` — CLI entrypoint.
- `triase_ml/` — Package code:
  - `data/` — Excel loading, **renaming**, cleaning.
  - `features/` — Feature selection methods.
  - `models/` — Model registry (sklearn, xgboost, lightgbm).
  - `training/` — CV runner, elimination, metrics, tuning.
  - `viz/` — Plots and SHAP visualizations.
- `param_grid.json` — Example per-model hyperparameter grid.
- `verification_param_grid.json` — Small grid for verification.
- `run_sweep.ps1` — Main sweep script.
- `run_verification.ps1` — **Verification script (Recommended)**.

## Installation

```bash
pip install -e .
```

Base dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`, `openpyxl`, `shap`.

Optional (for specific models/selectors):
```bash
pip install xgboost lightgbm
```

## Data & Column Renaming
The CLI loads `Triase_cleaned.xlsx`. **Columns are automatically renamed** from Indonesian to concise English identifiers internally.
Examples:
- `Tekanan darah sistolik` -> `bp_sys`
- `Usia (tahun)` -> `age`
- `Diagnosa Penyakit jantung pasien (text)` -> `diag_text`

## Tasks (`--task`)
- `diagnosis` — Predict diagnosis (`y_diagnosis`) from `X_no_diagnosis`.
- `handling_no_diag` — Predict handling (`y_handling`) from `X_no_diagnosis`.
- `handling_with_diag` — Predict handling (`y_handling`) from `X_with_diagnosis`.
- `pipeline_diag_then_handling` — 2-stage pipeline:
  1. Train diagnosis model using tuned parameters.
  2. Overwrite diagnosis columns in `X` with Stage 1 predictions.
  3. Train handling model using tuned parameters.

## Models (`--models`)
Supports comma-separated list or `all`.
- `logreg`, `knn`, `lda`, `mlp`, `svm_linear`, `svm_rbf`
- **Trees**: `rf` (ExtraTrees), `extra_trees`, `gb`, `hgb`, `adaboost`, `bagging`
- **External**: `xgb`, `lgbm` (require extra packages)
- **Ensembles**:
    - `voting`: Uses **soft voting** (probabilities) to enable ROC AUC graphs.
    - `stacking`

## Usage Examples

### 1. Verify Pipeline (Run this first)
Runs all models with a minimal grid to ensure everything works.
```powershell
.\run_verification.ps1
```

### 2. Diagnosis (10-fold CV)
```bash
python run_triase.py --excel Triase_cleaned.xlsx --task diagnosis \
  --models rf,xgb,svm_rbf \
  --feature_method mutual_info --top_n 20 \
  --cv stratified_kfold --n_splits 10 \
  --out outputs/diagnosis_mi20
```

### 3. Handling (Tuned)
```bash
python run_triase.py --excel Triase_cleaned.xlsx --task handling_with_diag \
  --models rf \
  --feature_method rf_importance --top_n 30 \
  --eliminate --agg median --tune --k_grid 0.3,0.5,0.7 \
  --param_grid_json param_grid.json \
  --out outputs/handling_tuned
```

## Outputs (`--out`)
- `results_summary.json`: Aggregated metrics.
- `tuning/<model>.json`: Best params found.
- `figures/<model>/`:
  - `confusion_matrix.png`
  - `roc_auc.png`
  - `shap_beeswarm.png` (Global SHAP summary on valid/test set)

## Known Notes
- **SHAP**: Uses the first CV split for generation. Now correctly uses **tuned model parameters**.
- **Manual Selector**: The keyword-based `manual` selector is available in code but not exposed in CLI choices by default.
