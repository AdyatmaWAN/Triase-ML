from __future__ import annotations

import pandas as pd

from .cleaning import build_datasets_from_dataframe, TriaseDatasets


def load_triase_excel(excel_path: str, sheet_name: str = "iccu_cleaned_imputed") -> TriaseDatasets:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    return build_datasets_from_dataframe(df)
