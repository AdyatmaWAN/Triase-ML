from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


CAT_COL_COLOR = "triage_color"
CAT_COL_GCS = "gcs"
CAT_COL_CONDITION = "general_condition"
CAT_COL_CONSCIOUS = "consciousness"
CAT_COL_EKG = "ecg_st"
CAT_COL_DIAG = "diag_text"

DROP_COLS_COMMON = [
    "ews",
    "oxygen_sat",
    "medication",
    "trop_i",
    "ckmb",
    "id",
    "date_iccu",
    "mrn",
    "ecg_desc",
    "final_diag_text",
    "primary_pci",
    "elective_pci",
    "cabg_cito",
    "cabg_elective",
]

HANDLING_COLS = ["primary_pci", "elective_pci"]


@dataclass(frozen=True)
class TriaseDatasets:
    # Features without diagnosis column (can be used for diagnosis OR handling_no_diag)
    X_no_diagnosis: pd.DataFrame
    y_diagnosis: pd.Series

    # Features with diagnosis column (used for handling_with_diag)
    X_with_diagnosis: pd.DataFrame
    y_handling: pd.Series


def _normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Color
    df[CAT_COL_COLOR] = df[CAT_COL_COLOR].replace([" ", "-"], np.nan)
    df[CAT_COL_COLOR] = df[CAT_COL_COLOR].replace(np.nan, "UnknownColor")
    df[CAT_COL_COLOR] = df[CAT_COL_COLOR].replace("merah", "Merah")
    df[CAT_COL_COLOR] = df[CAT_COL_COLOR].replace("kuning", "Kuning")
    df[CAT_COL_COLOR] = df[CAT_COL_COLOR].replace("hijau", "Hijau")

    # Condition
    df[CAT_COL_CONDITION] = df[CAT_COL_CONDITION].replace("sedang", "Sedang")
    df[CAT_COL_CONDITION] = df[CAT_COL_CONDITION].replace("Sedang", "TSS")
    df[CAT_COL_CONDITION] = df[CAT_COL_CONDITION].replace("berat", "TSB")
    df[CAT_COL_CONDITION] = df[CAT_COL_CONDITION].replace("baik", "TSR")

    # Consciousness
    df[CAT_COL_CONSCIOUS] = df[CAT_COL_CONSCIOUS].replace("Sadar", "Compos mentis ")
    df[CAT_COL_CONSCIOUS] = df[CAT_COL_CONSCIOUS].replace("Compos mentis ", "Compos mentis")
    df[CAT_COL_CONSCIOUS] = df[CAT_COL_CONSCIOUS].replace("penurunan kesadaran", "Delirium")

    # EKG formatting
    df[CAT_COL_EKG] = df[CAT_COL_EKG].replace("STE ", "STE")
    df[CAT_COL_EKG] = df[CAT_COL_EKG].replace("STN ", "STN")

    return df


def _derive_handling_target(df: pd.DataFrame) -> pd.Series:
    # 1 = primary_pci, 2 = elective_pci, 0 = none
    handling = df.apply(
        lambda row: 1 if row.get("primary_pci", 0) == 1
        else 2 if row.get("elective_pci", 0) == 1
        else 0,
        axis=1,
    )
    return handling.astype(int)


def build_datasets_from_dataframe(df: pd.DataFrame) -> TriaseDatasets:
    df = _normalize_categoricals(df)

    y_diagnosis = df[CAT_COL_DIAG].copy()
    y_handling = _derive_handling_target(df)

    categorical_no_diag: List[str] = [CAT_COL_COLOR, CAT_COL_GCS, CAT_COL_CONDITION, CAT_COL_CONSCIOUS, CAT_COL_EKG]
    categorical_with_diag: List[str] = categorical_no_diag + [CAT_COL_DIAG]

    # without diagnosis in X
    X_no_diag = pd.get_dummies(df, columns=categorical_no_diag, dtype=int)
    X_no_diag = X_no_diag.drop(columns=DROP_COLS_COMMON + [CAT_COL_DIAG], errors="ignore")

    # with diagnosis in X
    X_with_diag = pd.get_dummies(df, columns=categorical_with_diag, dtype=int)
    X_with_diag = X_with_diag.drop(columns=DROP_COLS_COMMON + ["handling"], errors="ignore")
    # remove the handling source columns from X_with_diag if still there
    X_with_diag = X_with_diag.drop(columns=HANDLING_COLS, errors="ignore")

    return TriaseDatasets(
        X_no_diagnosis=X_no_diag,
        y_diagnosis=y_diagnosis,
        X_with_diagnosis=X_with_diag,
        y_handling=y_handling,
    )
