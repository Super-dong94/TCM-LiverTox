from __future__ import annotations

from typing import Any

import pandas as pd


IDENTIFIER_COLUMNS = ("CID", "ChemicalName", "Smiles")
REFERENCE_OUTPUT_COLUMNS = (
    "Bioavailability_Ma",
    "Reference_Match",
    "IntoBlood",
    "IntoBlood_Reason",
)


def _match_key(column: str, value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if column == "CID":
        try:
            return str(int(float(text)))
        except (TypeError, ValueError):
            return None
    if column == "ChemicalName":
        return text.casefold()
    return text


def match_intoblood_reference(
    query_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """Match compounds by CID, ChemicalName, or Smiles and copy reference flags."""
    if "IntoBlood" not in reference_df.columns:
        raise ValueError("入血参考库缺少 IntoBlood 列。")

    shared_identifiers = [
        column
        for column in IDENTIFIER_COLUMNS
        if column in query_df.columns and column in reference_df.columns
    ]
    if not shared_identifiers:
        raise ValueError("输入化合物与入血参考库没有可用的共同标识列。")

    reference = reference_df.reset_index(drop=True)
    lookups: dict[str, dict[str, int]] = {}
    for column in shared_identifiers:
        lookup: dict[str, int] = {}
        for index, value in reference[column].items():
            key = _match_key(column, value)
            if key is not None and key not in lookup:
                lookup[key] = int(index)
        lookups[column] = lookup

    output = query_df.reset_index(drop=True).copy()
    output["Bioavailability_Ma"] = pd.NA
    output["Reference_Match"] = False
    output["IntoBlood"] = 0
    output["IntoBlood_Reason"] = "not_in_reference"

    matched = 0
    for output_index, row in output.iterrows():
        reference_index = None
        for column in shared_identifiers:
            key = _match_key(column, row.get(column))
            if key is not None and key in lookups[column]:
                reference_index = lookups[column][key]
                break
        if reference_index is None:
            continue
        matched += 1
        reference_row = reference.loc[reference_index]
        for column in REFERENCE_OUTPUT_COLUMNS:
            if column in reference.columns:
                output.at[output_index, column] = reference_row[column]

    output["Bioavailability_Ma"] = pd.to_numeric(output["Bioavailability_Ma"], errors="coerce")
    output["Reference_Match"] = output["Reference_Match"].fillna(False).astype(bool)
    output["IntoBlood"] = pd.to_numeric(output["IntoBlood"], errors="coerce").fillna(0).astype(int)
    output["IntoBlood_Reason"] = output["IntoBlood_Reason"].fillna("not_in_reference").astype(str)

    if verbose:
        print(f"入血参考库精确匹配完成：{matched}/{len(output)} 个化合物命中。")
    return output
