param(
    [string]$PYTHON = "python"
)

# Non-GUI backend untuk plot (hindari Tkinter issues)
$env:MPLBACKEND = "Agg"

Write-Host "Using Python:" $PYTHON
& $PYTHON -c "import sys; print('Python:', sys.executable)"
& $PYTHON -c "import sklearn, pandas, numpy; print('core imports ok')" | Out-Null

$EXCEL   = "Triase_cleaned.xlsx"
$OUTROOT = "outputs\sweep_shap_debug"
$NSPLITS = 3       # LIGHT: 3-fold untuk debugging
$TOPN    = 15
$TASKS   = @("diagnosis")

# Selector list (hapus "manual" jika belum diimplementasikan)
$SELECTORS = @("none","xgb_gain","rf_importance","lgbm_importance","lasso_coef","mutual_info","chi2","rfe_rf","sfs_rf")
# Kalau Anda tetap mau manual tapi belum ada, uncomment ini dan comment list di atas:
# $SELECTORS = @("none","xgb_gain","rf_importance","lgbm_importance","lasso_coef","mutual_info","chi2","rfe_rf","sfs_rf","manual")

# Eliminate setting untuk test SHAP (sesuai contoh Anda)
$ELIMMODES = @("mean")
$KLIST = @("0.3")

# Semua model yang ada di registry (sesuaikan jika registry berubah)
$MODELS = @(
  "logreg","rf","knn","gb","hgb","extra_trees","adaboost","bagging",
  "xgb","lgbm","mlp","svm_linear","svm_rbf","lda","voting","stacking"
)

foreach ($task in $TASKS) {
  foreach ($sel in $SELECTORS) {
    foreach ($emode in $ELIMMODES) {
      foreach ($k in $KLIST) {
        foreach ($m in $MODELS) {

          $outDir = Join-Path $OUTROOT (Join-Path $task (Join-Path ("selector_" + $sel) (Join-Path ("elim_" + $emode + "_k_" + $k) ("model_" + $m))))
          New-Item -ItemType Directory -Force -Path $outDir | Out-Null

          Write-Host "RUN: task=$task sel=$sel elim=$emode k=$k model=$m"

          & $PYTHON run_triase.py `
            --excel $EXCEL `
            --task $task `
            --models $m `
            --feature_method $sel `
            --top_n $TOPN `
            --cv stratified_kfold `
            --n_splits $NSPLITS `
            --eliminate `
            --agg $emode `
            --k $k `
            --out $outDir `
            --no_save_models
        }
      }
    }
  }
}

Write-Host "DONE. Outputs in: $OUTROOT"
