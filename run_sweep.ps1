param(
    [string]$PYTHON = "python"
)

Write-Host "Using Python:" $PYTHON

$EXCEL   = "Triase_cleaned.xlsx"
$OUTROOT = "outputs\sweep_all"
$NSPLITS = 10

$TASKS = @("diagnosis","handling_no_diag","handling_with_diag","pipeline_diag_then_handling")
$SELECTORS = @("none","xgb_gain","rf_importance","lgbm_importance","lasso_coef","mutual_info","chi2","rfe_rf","sfs_rf","manual")
$ELIMMODES = @("none","median","mean")

$KGRID = "0.3,0.5,0.7"

foreach ($task in $TASKS) {
  foreach ($sel in $SELECTORS) {
    foreach ($emode in $ELIMMODES) {

      $outDir = Join-Path $OUTROOT (Join-Path $task (Join-Path ("selector_" + $sel) ("elim_" + $emode)))
      New-Item -ItemType Directory -Force -Path $outDir | Out-Null

      $baseArgs = @(
        "run_triase.py",
        "--excel", $EXCEL,
        "--task", $task,
        "--models", "all",
        "--feature_method", $sel,
        "--cv", "stratified_kfold",
        "--n_splits", $NSPLITS,
        "--tune",
        "--k_grid", $KGRID,
        "--tune_metric", "macro_f1",
        "--param_grid_json", "param_grid.json",
        "--out", $outDir
      )

      if ($emode -eq "none") {
        Write-Host "RUN: task=$task sel=$sel elim=none"
        & $PYTHON @baseArgs
      } else {
        Write-Host "RUN: task=$task sel=$sel elim=$emode"
        & $PYTHON @baseArgs --eliminate --agg $emode
      }
    }
  }
}

Write-Host "DONE. Outputs in: $OUTROOT"
