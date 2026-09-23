# %%
import os
import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np

# RDKit 相关
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit import RDLogger

if __package__:
    from .model_runtime import load_model_bundles, predict_multimodel_pipeline as predict_model_bundles, standardize_smiles, ENDPOINT_THRESHOLDS, METHOD_VERSION
    from .intoblood_pred import assess_blood_exposure
else:
    from model_runtime import load_model_bundles, predict_multimodel_pipeline as predict_model_bundles, standardize_smiles, ENDPOINT_THRESHOLDS, METHOD_VERSION
    from intoblood_pred import assess_blood_exposure



def resource_root() -> Path:
    override = os.getenv("TOXHERB_RESOURCE_ROOT")
    if override:
        return Path(override).resolve()
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root).resolve()
    return Path(__file__).resolve().parents[1]

RDLogger.DisableLog('rdApp.*')

# =======================================================
# 多模型加载与预测管线
# =======================================================
def load_models():
    model_dir = resource_root() / "models"
    model_paths = (
        Path(os.getenv("TOXHERB_MODEL_CELL", str(model_dir / "01best_model_tuned.joblib"))),
        Path(os.getenv("TOXHERB_MODEL_ANIMAL", str(model_dir / "02best_model_tuned.joblib"))),
        Path(os.getenv("TOXHERB_MODEL_CLINICAL", str(model_dir / "03best_model_tuned.joblib"))),
    )
    missing = [path for path in model_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model files: " + ", ".join(map(str, missing)))
    return load_model_bundles(model_paths)

def predict_multimodel_pipeline(smiles_list, models):
    return predict_model_bundles(smiles_list, models)

# =======================================================
# 3. 核心大循环：方剂整体评价系统
# =======================================================
def _normalize_herb_key(value):
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def _split_herb_names(value):
    if value is None or pd.isna(value):
        return []
    pieces = re.split(r"[\n,，;；、/|]+", str(value))
    return [piece.strip() for piece in pieces if piece and piece.strip()]


def _read_food_medicine_homology():
    path = resource_root() / "data" / "13_food_medicine_homology.csv"
    if not path.exists():
        return pd.DataFrame(), path
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gbk")
    if "HerbName" not in df.columns and "Herb.Chinese.name" in df.columns:
        df = df.copy()
        df["HerbName"] = df["Herb.Chinese.name"]
    return df, path


def _food_medicine_summary(herb_candidates):
    homology, path = _read_food_medicine_homology()
    source = "data/13_food_medicine_homology.csv"
    if homology.empty or "HerbName" not in homology.columns:
        return {
            "Food_medicine_homology_matched": False,
            "Food_medicine_homology_matched_herbs": "",
            "Food_medicine_homology_source": source,
            "Food_medicine_homology_note": f"药食同源名单不可用或为空: {path}",
        }
    candidate_keys = {_normalize_herb_key(value) for value in herb_candidates}
    candidate_keys.discard("")
    homology = homology.copy()
    homology["_HerbKey"] = homology["HerbName"].map(_normalize_herb_key)
    matched = homology[homology["_HerbKey"].isin(candidate_keys)]
    matched_herbs = sorted(
        matched["HerbName"].dropna().astype(str).map(_normalize_herb_key).replace("", np.nan).dropna().unique().tolist()
    )
    if matched_herbs:
        note = "命中药食同源名单，仅作为研究参考标注，不改变肝毒性预测结果。"
    elif herb_candidates:
        note = "未命中药食同源名单，肝毒性预测结果照常供研究者参考。"
    else:
        note = "未识别到可用于药食同源匹配的药材名称。"
    return {
        "Food_medicine_homology_matched": bool(matched_herbs),
        "Food_medicine_homology_matched_herbs": "、".join(matched_herbs),
        "Food_medicine_homology_source": source,
        "Food_medicine_homology_note": note,
    }


def build_summary_annotations(livertox_df, herb_candidates=None, absorbed_molecule_count_or_percent=None):
    candidates = []
    for value in herb_candidates or []:
        candidates.extend(_split_herb_names(value))
    annotations = {}
    annotations.update(_food_medicine_summary(candidates))
    annotations["Method_version"] = METHOD_VERSION
    annotations["Blood_exposure_rule"] = "Reference_Match OR Bioavailability_Ma >= 0.3"
    annotations["Cell_threshold"] = ENDPOINT_THRESHOLDS["cell"]
    annotations["Animal_threshold"] = ENDPOINT_THRESHOLDS["animal"]
    annotations["Clinical_threshold"] = ENDPOINT_THRESHOLDS["clinical"]
    annotations["Candidate_rule"] = "Hepatotoxicity_Score >= 1"
    annotations["Count_basis"] = "unique CID; standardized SMILES when CID is missing"
    return annotations


def compound_keys(df):
    cid = pd.to_numeric(df.get("CID", pd.Series(float("nan"), index=df.index)), errors="coerce")
    structures = df["Standardized_SMILES"].fillna("") if "Standardized_SMILES" in df else df.get("Smiles", pd.Series("", index=df.index)).map(standardize_smiles).fillna("")
    return pd.Series([f"CID:{int(c)}" if pd.notna(c) else (f"SMILES:{s}" if s else f"unresolved:{i}") for i, (c, s) in enumerate(zip(cid, structures))], index=df.index, dtype="str")


def candidate_target_links(candidates, chemical_names, target_relations):
    meta_cols = [c for c in ["CID", "ChemicalName", "Smiles", "Standardized_SMILES", "Compound_Key", "Source_Herbs", "Source_Formulas", "Herb.Chinese.name", "Formula.Chinese.name"] if c in candidates]
    columns = list(dict.fromkeys(meta_cols + ["ChemicalName", "Symbol"]))
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    source = candidates[meta_cols].copy()
    source["CID"] = pd.to_numeric(source.get("CID"), errors="coerce")
    if "ChemicalName" not in source:
        source["ChemicalName"] = None
    source = source.rename(columns={"ChemicalName": "_candidate_name"})
    aliases = chemical_names[["CID", "ChemicalName"]].copy()
    aliases["CID"] = pd.to_numeric(aliases["CID"], errors="coerce")
    aliases = aliases[aliases["CID"].notna()].drop_duplicates()
    with_cid = source[source["CID"].notna()].merge(aliases, on="CID", how="left")
    without_cid = source[source["CID"].isna()].copy()
    without_cid["ChemicalName"] = without_cid["_candidate_name"]
    links = pd.concat([with_cid, without_cid], ignore_index=True)
    links["ChemicalName"] = links["ChemicalName"].fillna(links["_candidate_name"])
    links = links.drop(columns="_candidate_name")
    return links.merge(target_relations[["ChemicalName", "Symbol"]].drop_duplicates(), on="ChemicalName", how="inner").drop_duplicates().reset_index(drop=True)


def count_table(df, column, count_type, top_n=None):
    columns = ["Count_Type", "Source_Column", "Item", "Count", "Raw_Record_Count", "Count_Basis"]
    if df.empty or column not in df:
        return pd.DataFrame(columns=columns)
    work = df.copy()
    is_target = column in ["TERM", "GOID", "GOTerm", "GOTermName", "PathwayName", "Pathwayid", "PathwayID"]
    if is_target:
        ids = work.get("ENTREZID", pd.Series("", index=work.index)).fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
        work["_entity"] = ids.mask(ids.eq(""), work.get("Symbol", pd.Series("", index=work.index))).fillna("")
    else:
        work["_entity"] = compound_keys(work)
        unresolved = work["_entity"].str.startswith("unresolved:")
        work.loc[unresolved, "_entity"] = work.get("ChemicalName", pd.Series("", index=work.index)).fillna("")[unresolved]
    work = work[work["_entity"].ne("") & work[column].notna()].copy()
    if column in ["Class", "Superclass", "Pathway"]:
        work[column] = work[column].astype(str).str.split(r"[;,；|]")
        work = work.explode(column)
    work[column] = work[column].astype(str).str.strip()
    work = work[work[column].ne("")]
    counts = work.groupby(column)["_entity"].nunique().sort_values(ascending=False, kind="stable")
    raw = work.groupby(column)["Raw_Record_Count"].sum() if "Raw_Record_Count" in work else work.groupby(column).size()
    rows = [{"Count_Type": count_type, "Source_Column": column, "Item": name, "Count": int(count),
             "Raw_Record_Count": int(raw[name]), "Count_Basis": "unique_target" if is_target else "unique_compound"} for name, count in counts.items()]
    result = pd.DataFrame(rows, columns=columns)
    return result.head(top_n) if top_n is not None else result


def add_endpoint_scores(df, thresholds=None):
    out = df.copy()
    thresholds = thresholds or ENDPOINT_THRESHOLDS
    columns = ["Pred_Cell_prob", "Pred_Animal_prob", "Pred_Clinical_prob"]
    probabilities = out.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    probabilities = probabilities.where(probabilities.ge(0) & probabilities.le(1))
    out[columns] = probabilities
    evaluated = probabilities.notna().all(axis=1)
    for endpoint, label in zip(ENDPOINT_THRESHOLDS, ["Cell", "Animal", "Clinical"]):
        out[f"Pred_{label}_Toxicity"] = probabilities[f"Pred_{label}_prob"].ge(thresholds[endpoint]).astype(float).where(evaluated)
    out["Hepatotoxicity_Score"] = out[["Pred_Cell_Toxicity", "Pred_Animal_Toxicity", "Pred_Clinical_Toxicity"]].sum(axis=1).where(evaluated)
    out["Max_Tox_Prob"] = probabilities.max(axis=1).where(evaluated)
    out["Positive_Endpoints"] = None
    out["Endpoint_Combination"] = None
    out["Pmax_Source"] = None
    for idx in out.index[evaluated]:
        positive = [endpoint for endpoint, label in zip(ENDPOINT_THRESHOLDS, ["Cell", "Animal", "Clinical"]) if out.at[idx, f"Pred_{label}_Toxicity"] == 1]
        out.at[idx, "Positive_Endpoints"] = ";".join(positive)
        out.at[idx, "Endpoint_Combination"] = "+".join(positive) or "none"
        out.at[idx, "Pmax_Source"] = ";".join(endpoint for endpoint, column in zip(ENDPOINT_THRESHOLDS, columns) if probabilities.at[idx, column] == out.at[idx, "Max_Tox_Prob"])
    return out


def evaluate_compounds(frame, models=None):
    df = assess_blood_exposure(frame).reset_index(drop=True)
    df["Standardized_SMILES"] = df["Smiles"].map(standardize_smiles)
    df["Compound_Key"] = compound_keys(df)
    df = df.drop_duplicates("Compound_Key").reset_index(drop=True)
    df["is_valid_smiles"] = df["Standardized_SMILES"].notna()
    for name, calculation, digits in [("LogP", Descriptors.MolLogP, 2), ("MW", Descriptors.MolWt, 2), ("QED", QED.qed, 3)]:
        df[name] = df["Standardized_SMILES"].map(lambda smi: round(float(calculation(Chem.MolFromSmiles(smi))), digits) if isinstance(smi, str) and smi else float("nan"))
    df["OB_Percent"] = df["Bioavailability_Ma"]
    df["Is_Absorbed"] = df["IntoBlood"].eq(1).fillna(False) & df["is_valid_smiles"]
    df["Assessment_Status"] = "not_evaluated"
    df.loc[df["IntoBlood"].eq(0).fillna(False), "Assessment_Status"] = "screened_out"
    df.loc[~df["is_valid_smiles"], "IntoBlood_Reason"] = "invalid_structure"
    df.loc[~df["is_valid_smiles"], "IntoBlood"] = pd.NA
    for label in ["Cell", "Animal", "Clinical"]:
        df[f"Pred_{label}_prob"] = float("nan")
    eligible = df.index[df["Is_Absorbed"]]
    if len(eligible):
        models = models if models is not None else load_models()
        unique_smiles = df.loc[eligible, "Standardized_SMILES"].drop_duplicates().tolist()
        outputs = predict_multimodel_pipeline(unique_smiles, models)
        for label, probabilities in zip(["Cell", "Animal", "Clinical"], outputs[1::2]):
            df.loc[eligible, f"Pred_{label}_prob"] = df.loc[eligible, "Standardized_SMILES"].map(dict(zip(unique_smiles, probabilities)))
        df.loc[eligible, "Assessment_Status"] = "evaluated"
    thresholds = {bundle["endpoint"]: round(float(bundle["threshold"]), 12) for bundle in models} if models is not None else ENDPOINT_THRESHOLDS
    return add_endpoint_scores(df, thresholds)


def prediction_summary(df):
    from itertools import product
    work = df.copy()
    work["Compound_Key"] = compound_keys(work)
    work = work.drop_duplicates("Compound_Key")
    evaluated = work[work["Hepatotoxicity_Score"].notna()]
    total, count = len(work), len(evaluated)
    probs = {endpoint: (float(evaluated[f"Pred_{label}_prob"].max()) if count else None) for endpoint, label in zip(ENDPOINT_THRESHOLDS, ["Cell", "Animal", "Clinical"])}
    probs["max"] = float(evaluated["Max_Tox_Prob"].max()) if count else None
    reasons = work.get("IntoBlood_Reason", pd.Series("missing_admet", index=work.index))
    missing_count = int(reasons.isin(["invalid_structure", "missing_admet", "not_in_reference"]).sum())
    candidate_count = int(evaluated["Hepatotoxicity_Score"].ge(1).sum())
    pmax_rows = evaluated[evaluated["Max_Tox_Prob"].eq(probs["max"])] if count else evaluated
    sources = [{"CID": None if pd.isna(row.get("CID")) else int(row["CID"]), "name": str(row.get("ChemicalName") or row.get("Compound_Key")), "endpoints": str(row["Pmax_Source"]).split(";")} for row in pmax_rows.to_dict("records")]
    summary = {
        "total_compounds": total, "valid_smiles_count": int(work.get("is_valid_smiles", work.get("Smiles", pd.Series("", index=work.index)).map(lambda value: isinstance(value, str) and Chem.MolFromSmiles(value) is not None)).sum()),
        "intoblood_count": int(work["IntoBlood"].eq(1).sum()), "absorbed_count": count, "evaluated_count": count,
        "not_evaluated_count": total - count, "insufficient_data_count": missing_count,
        "screened_out_count": int(reasons.eq("admet_below_0.3").sum()),
        "reference_match_count": int(reasons.eq("reference_match").sum()), "admet_pass_count": int(reasons.eq("admet_ge_0.3").sum()),
        "high_toxic_count": candidate_count, "candidate_count": candidate_count,
        "candidate_ratio": candidate_count / count if count else None,
        "max_toxic_probability": probs["max"], "pmax_contributors": sources,
        "assessment_status": "not_evaluable" if not count else ("partial" if missing_count else "completed"),
        "risk_level": "unassessed" if not count else "screening",
        "risk_label": "无法评估" if not count else ("部分完成" if missing_count else "筛查完成"),
        "method_version": METHOD_VERSION, "endpoint_thresholds": ENDPOINT_THRESHOLDS,
        "candidate_rule": "Hepatotoxicity_Score >= 1", "count_basis": "unique_compounds",
        "advice": "Pmax 为成分集合中的最大模型输出，0–3 分表示阳性端点数量；均不代表临床肝损伤发生率或安全性结论。",
    }
    endpoints = list(ENDPOINT_THRESHOLDS)
    combinations = ["+".join(e for e, bit in zip(endpoints, bits) if bit) or "none" for bits in product([0, 1], repeat=3)]
    statistics = {
        "endpoint_counts": [{"endpoint": e, "positive": int(evaluated[f"Pred_{label}_Toxicity"].eq(1).sum()), "negative": int(evaluated[f"Pred_{label}_Toxicity"].eq(0).sum())} for e, label in zip(endpoints, ["Cell", "Animal", "Clinical"])],
        "combination_counts": [{"name": name, "count": int(evaluated["Endpoint_Combination"].eq(name).sum())} for name in combinations],
        "score_counts": [{"score": score, "count": int(evaluated["Hepatotoxicity_Score"].eq(score).sum())} for score in range(4)],
        "pmax_source_counts": [{"name": name, "count": int(evaluated["Pmax_Source"].eq(name).sum())} for name in endpoints] + [{"name": "tied", "count": int(evaluated["Pmax_Source"].fillna("").astype(str).str.contains(";", na=False).sum())}],
    }
    return summary, probs, statistics


def evaluate_tcm_formula(csv_path, smiles_col="Smiles"):
    frame = pd.read_csv(csv_path).rename(columns={smiles_col: "Smiles"})
    df = evaluate_compounds(frame)
    valid_count = int(df["is_valid_smiles"].sum())
    evaluated_count = int(df["Assessment_Status"].eq("evaluated").sum())
    candidate_count = int(df["Hepatotoxicity_Score"].ge(1).sum())
    ratio = candidate_count / evaluated_count if evaluated_count else None
    label = "筛查完成" if evaluated_count else "无法评估"
    advice = "Pmax 为成分集合中的最大模型输出，0–3 分表示阳性端点数量；均不代表临床肝损伤发生率或安全性结论。"
    return df, valid_count, evaluated_count, 1, candidate_count, ratio, label, advice


if __name__ == "__main__":
    evaluate_tcm_formula("L6810-TCM-2928.csv")
