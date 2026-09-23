from __future__ import annotations

from typing import Any

import pandas as pd
from rdkit import Chem


IDENTIFIER_COLUMNS = ("CID", "ChemicalName", "Smiles")
REFERENCE_OUTPUT_COLUMNS = (
    "Bioavailability_Ma",
    "Reference_Match",
    "IntoBlood",
    "IntoBlood_Reason",
    "Precomputed_Match",
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


def assess_blood_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """实测匹配优先；预计算 ADMET >= 0.3，缺失与阴性分别保留。"""
    out = df.copy()
    ref = out.get("Reference_Match", pd.Series(False, index=out.index))
    out["Reference_Match"] = ref.fillna(False).astype(str).str.lower().isin(["true", "1", "1.0"])
    out["Bioavailability_Ma"] = pd.to_numeric(out.get("Bioavailability_Ma", pd.Series(float("nan"), index=out.index)), errors="coerce")
    ob = out["Bioavailability_Ma"].where(out["Bioavailability_Ma"].between(0, 1))
    out["Bioavailability_Ma"] = ob
    ref = out["Reference_Match"]
    out["IntoBlood"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["IntoBlood_Reason"] = "missing_admet"
    out.loc[~ref & ob.notna(), "IntoBlood"] = ob[~ref & ob.notna()].ge(0.3).astype(int)
    out.loc[~ref & ob.ge(0.3), "IntoBlood_Reason"] = "admet_ge_0.3"
    out.loc[~ref & ob.lt(0.3), "IntoBlood_Reason"] = "admet_below_0.3"
    out.loc[ref, ["IntoBlood", "IntoBlood_Reason"]] = [1, "reference_match"]
    if "Smiles" in out:
        valid = out["Smiles"].map(lambda value: isinstance(value, str) and bool(value.strip()) and Chem.MolFromSmiles(value) is not None)
        out.loc[~valid, "IntoBlood"] = pd.NA
        out.loc[~valid, "IntoBlood_Reason"] = "invalid_structure"
    return out


def match_intoblood_reference(
    query_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """匹配预计算数据；Reference_Match 仅表示原实测参考库命中。"""
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
    output["Precomputed_Match"] = False
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
        output.at[output_index, "Precomputed_Match"] = True

    output = assess_blood_exposure(output)

    if verbose:
        print(f"入血参考库精确匹配完成：{matched}/{len(output)} 个化合物命中。")
    return output
