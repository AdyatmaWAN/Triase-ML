#!/usr/bin/env bash
set -euo pipefail

EXCEL="Triase_cleaned.xlsx"
OUT_ROOT="outputs/sweep_all"
NSPLITS=10

TASKS=("diagnosis" "handling_no_diag" "handling_with_diag" "pipeline_diag_then_handling")
SELECTORS=("none" "xgb_gain" "rf_importance" "lgbm_importance" "lasso_coef" "mutual_info" "chi2" "rfe_rf" "sfs_rf" "manual")

# Eliminate modes:
# 1) no eliminate
# 2) eliminate median
# 3) eliminate mean
ELIM_MODES=("none" "median" "mean")

KGRID="0.3,0.5,0.7"

for task in "${TASKS[@]}"; do
  for sel in "${SELECTORS[@]}"; do
    for emode in "${ELIM_MODES[@]}"; do

      out_dir="${OUT_ROOT}/${task}/selector_${sel}/elim_${emode}"
      mkdir -p "${out_dir}"

      base_cmd=(
        python run_triase.py
        --excel "${EXCEL}"
        --task "${task}"
        --models all
        --feature_method "${sel}"
        --cv stratified_kfold
        --n_splits "${NSPLITS}"
        --tune
        --k_grid "${KGRID}"
        --tune_metric macro_f1
        --param_grid param_grid.json
        --out "${out_dir}"
      )

      if [[ "${emode}" == "none" ]]; then
        echo "RUN: task=${task} sel=${sel} elim=none"
        "${base_cmd[@]}"
      else
        echo "RUN: task=${task} sel=${sel} elim=${emode}"
        "${base_cmd[@]}" --eliminate --agg "${emode}"
      fi

    done
  done
done

echo "DONE. Outputs in: ${OUT_ROOT}"
