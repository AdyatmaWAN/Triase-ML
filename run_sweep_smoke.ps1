param(
    [string]$PYTHON = "python"
)

# Non-GUI backend untuk plot (hindari Tkinter issues)
$env:MPLBACKEND = "Agg"

Write-Host "Using Python:" $PYTHON
& $PYTHON -c "import sklearn, pandas, numpy; print('core imports ok')" | Out-Null

$EXCEL   = "Triase_cleaned.xlsx"
$OUTROOT = "outputs\sweep_smoke"
$NSPLITS = 3     

$TASKS = @("diagnosis")
$SELECTORS = @("manual", "none","xgb_gain","rf_importance","lgbm_importance","lasso_coef","mutual_info","chi2","rfe_rf","sfs_rf")
$ELIMMODES = @("mean")
$KLIST = @("0.3")

$MODELS = "all"

foreach ($task in $TASKS) {
  foreach ($sel in $SELECTORS) {
    foreach ($emode in $ELIMMODES) {

      if ($emode -eq "none") {
        $outDir = Join-Path $OUTROOT (Join-Path $task (Join-Path ("selector_" + $sel) "elim_none"))
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null

        Write-Host "RUN: task=$task sel=$sel elim=none"
        & $PYTHON run_triase.py `
          --excel $EXCEL `
          --task $task `
          --models $MODELS `
          --feature_method $sel `
          --top_n 15 `
          --cv stratified_kfold `
          --n_splits $NSPLITS `
          --out $outDir `
          --no_save_models
      }
      else {
        foreach ($k in $KLIST) {
          $outDir = Join-Path $OUTROOT (Join-Path $task (Join-Path ("selector_" + $sel) ("elim_" + $emode + "_k_" + $k)))
          New-Item -ItemType Directory -Force -Path $outDir | Out-Null

          Write-Host "RUN: task=$task sel=$sel elim=$emode k=$k"
          & $PYTHON run_triase.py `
            --excel $EXCEL `
            --task $task `
            --models $MODELS `
            --feature_method $sel `
            --top_n 15 `
            --cv stratified_kfold `
            --n_splits $NSPLITS `
            --eliminate `
            --agg $emode `
            --k $k `
            --out $outDir `
#             --no_save_models
        }
      }
    }
  }
}

Write-Host "DONE. Outputs in: $OUTROOT"
