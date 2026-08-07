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

from model_runtime import (
    load_model_bundles,
    predict_multimodel_pipeline as predict_model_bundles,
)


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


def _absorbed_count_value(value, fallback):
    if value is None or pd.isna(value):
        return fallback
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else fallback


def _risk_from_enrichment(enrichment, absorbed_count):
    if absorbed_count == 0:
        return "🟢 低风险 (Safe)", "未检出可用于模型一致性参考的有效吸收成分。"
    if enrichment < 0.02:
        return "🟢 低风险 (Safe)", "模型一致性参考提示整体肝毒性风险较低。"
    if enrichment < 0.05:
        return "🟡 中等风险 (Moderate Risk)", "模型一致性参考提示存在肝毒性成分富集迹象。"
    return "🔴 高风险 (High Risk)", "模型一致性参考提示系统性毒性富集率超过阈值。"


def _consensus_summary(livertox_df, absorbed_molecule_count_or_percent=None):
    threshold = 0.75
    empty = {
        "Consensus_max_probability": 0.0,
        "Consensus_high_toxic_molecule_count": 0,
        "Consensus_enrichment_ratio": 0.0,
        "Consensus_risk_level": "🟢 低风险 (Safe)",
        "Consensus_advice": "未检出可用于模型一致性参考的有效吸收成分。",
        "Consensus_toxic_probability_threshold": threshold,
        "Consensus_evaluated_molecule_count": 0,
        "Animal_probability_saturation_ratio": 0.0,
        "Model_quality_flags": "",
    }
    if livertox_df.empty:
        return empty

    work = livertox_df.copy()
    if "Is_Absorbed" in work.columns:
        absorbed_mask = work["Is_Absorbed"].map(
            lambda value: value is True or str(value).strip().casefold() in {"true", "1", "yes"}
        )
        work = work[absorbed_mask].copy()
    if work.empty:
        return empty

    index = work.index
    weighted_sum = pd.Series(0.0, index=index)
    weight_sum = pd.Series(0.0, index=index)

    def add_weighted(column, weight, calibrated=False):
        if column not in work.columns:
            return
        values = pd.to_numeric(work[column], errors="coerce")
        values = values.where(values >= 0)
        if calibrated:
            values = (0.5 + 0.35 * (values - 0.5)).clip(lower=0.0, upper=1.0)
        valid = values.notna()
        weighted_sum.loc[valid] += values.loc[valid] * weight
        weight_sum.loc[valid] += weight

    add_weighted("Pred_Cell_prob", 0.35)
    add_weighted("Pred_Animal_prob", 0.25, calibrated=True)
    add_weighted("Pred_Clinical_prob", 0.40)

    consensus = (weighted_sum / weight_sum).where(weight_sum > 0).dropna()
    if consensus.empty:
        return empty

    evaluated_count = int(len(consensus))
    denominator = _absorbed_count_value(absorbed_molecule_count_or_percent, evaluated_count)
    if denominator <= 0:
        denominator = evaluated_count
    high_count = int((consensus >= threshold).sum())
    enrichment = high_count / denominator if denominator else 0.0
    risk_level, advice = _risk_from_enrichment(enrichment, denominator)

    animal_raw = pd.to_numeric(work.get("Pred_Animal_prob", pd.Series(dtype=float)), errors="coerce")
    animal_raw = animal_raw[animal_raw >= 0]
    saturation_ratio = float((animal_raw >= 0.99).sum() / len(animal_raw)) if len(animal_raw) else 0.0
    flags = ["animal_probability_saturation"] if saturation_ratio > 0.8 else []

    return {
        "Consensus_max_probability": round(float(consensus.max()), 4),
        "Consensus_high_toxic_molecule_count": high_count,
        "Consensus_enrichment_ratio": round(float(enrichment), 6),
        "Consensus_risk_level": risk_level,
        "Consensus_advice": advice,
        "Consensus_toxic_probability_threshold": threshold,
        "Consensus_evaluated_molecule_count": evaluated_count,
        "Animal_probability_saturation_ratio": round(saturation_ratio, 6),
        "Model_quality_flags": "、".join(flags),
    }


def build_summary_annotations(livertox_df, herb_candidates=None, absorbed_molecule_count_or_percent=None):
    candidates = []
    for value in herb_candidates or []:
        candidates.extend(_split_herb_names(value))
    annotations = {}
    annotations.update(_food_medicine_summary(candidates))
    annotations.update(_consensus_summary(livertox_df, absorbed_molecule_count_or_percent))
    return annotations


def evaluate_tcm_formula(csv_path, smiles_col='Smiles'):
    # print(f"\n[{csv_path}] 开始全景跨维度肝毒性评估...")
    df = pd.read_csv(csv_path)

    df['is_valid'] = df[smiles_col].apply(lambda x: pd.notna(x) and Chem.MolFromSmiles(x) is not None)
    df = df[df['is_valid']].drop(columns=['is_valid']).reset_index(drop=True)

    if len(df) == 0:
        return print("未找到有效的SMILES结构，分析终止。")
    com_num = len(df)
    df['LogP'] = df[smiles_col].apply(lambda smi: round(Descriptors.MolLogP(Chem.MolFromSmiles(smi)), 2))
    df['MW'] = df[smiles_col].apply(lambda smi: round(Descriptors.MolWt(Chem.MolFromSmiles(smi)), 2))
    df['QED'] = df[smiles_col].apply(lambda smi: round(QED.qed(Chem.MolFromSmiles(smi)), 3))
    if "Bioavailability_Ma" not in df.columns:
        raise ValueError("入血判定结果缺少 Bioavailability_Ma 列。")
    df['OB_Percent'] = pd.to_numeric(df["Bioavailability_Ma"], errors="coerce")

    # 阶段 B：联合入血过滤
    pass_flags = []
    for _, row in df.iterrows():
        pass_pc = (
            row.get("IntoBlood") == 1
            and row['LogP'] <= 5.0
            and row['MW'] <= 500
            and row['QED'] >= 0.3
            and pd.notna(row['OB_Percent'])
            and row['OB_Percent'] > 0.3
        )
        pass_flags.append(pass_pc)

    df['Is_Absorbed'] = pass_flags
    absorbed_mols_count = sum(pass_flags)
    # print(f"-> 符合联合标准 (入血) 分子数: {absorbed_mols_count}")

    # 【核心】：初始化你要的 6 列输出，未入血成分全部标记为 -1
    df['Pred_Cell_Toxicity'] = -1
    df['Pred_Cell_prob'] = -1.0

    df['Pred_Animal_Toxicity'] = -1
    df['Pred_Animal_prob'] = -1.0

    df['Pred_Clinical_Toxicity'] = -1
    df['Pred_Clinical_prob'] = -1.0

    df['Max_Tox_Prob'] = -1.0
    TOXIC_THRESHOLD = 0.85
    high_toxic_count = 0
    enrichment_ratio = 0.0
    risk_level = "🟢 低风险 (Safe)"
    advise = "未检出满足入血与吸收条件的有效成分，建议结合实验数据复核。"

    if absorbed_mols_count > 0:
        # print("-> 正在启动多模型联合矩阵预测...")
        models = load_models()

        absorbed_indices = df[df['Is_Absorbed'] == True].index
        absorbed_smiles = df.loc[absorbed_indices, smiles_col].tolist()

        # 接收这 6 个返回值
        (pred_cell, prob_cell,
         pred_animal, prob_animal,
         pred_clinical, prob_clinical) = predict_multimodel_pipeline(absorbed_smiles, models)

        # 精准回填到 DataFrame 对应的行
        df.loc[absorbed_indices, 'Pred_Cell_Toxicity'] = pred_cell
        df.loc[absorbed_indices, 'Pred_Cell_prob'] = prob_cell

        df.loc[absorbed_indices, 'Pred_Animal_Toxicity'] = pred_animal
        df.loc[absorbed_indices, 'Pred_Animal_prob'] = prob_animal

        df.loc[absorbed_indices, 'Pred_Clinical_Toxicity'] = pred_clinical
        df.loc[absorbed_indices, 'Pred_Clinical_prob'] = prob_clinical

        # 计算该分子的“最大潜在毒性概率”用于系统评估
        df.loc[absorbed_indices, 'Max_Tox_Prob'] = np.max([prob_cell, prob_animal, prob_clinical], axis=0)

        high_toxic_count = (df['Max_Tox_Prob'] >= TOXIC_THRESHOLD).sum()
        enrichment_ratio = high_toxic_count / absorbed_mols_count

        print("\n" + "="*50)
        print(" 📊 方剂肝毒性系统生物学评估报告")
        print("="*50)
        print(f"▶ 原始检出有效分子数: {com_num}")
        print(f"▶ 最终有效入血分子数: {absorbed_mols_count} (占总比 {absorbed_mols_count/com_num:.1%})")
        print(f"▶ 高危肝毒分子数量 (任一层级 P >= {TOXIC_THRESHOLD}): {high_toxic_count}")
        print(f"▶ 毒性成分体内综合富集率: {enrichment_ratio:.2%}")

        if enrichment_ratio < 0.02:
            risk_level = "🟢 低风险 (Safe)"
            advise = "方剂整体肝毒性风险极低，可常规使用。"
        elif 0.02 <= enrichment_ratio < 0.05:
            risk_level = "🟡 中等风险 (Moderate Risk)"
            advise = "存在肝毒性成分富集迹象，建议控制用量或辩证配伍。"
        else:
            risk_level = "🔴 高风险 (High Risk)"
            advise = "系统性毒性富集率严重超标！存在多层级验证的药物性肝损伤(DILI)高风险。"

    else:
        print("\n🟢 所有成分均被体内屏障过滤，系统判定为安全。")

    print(f"▶ 整体评价: {risk_level}")
    if absorbed_mols_count > 0:
        print(f"▶ 综合建议: {advise}")
    # 返回数据集，原始检出有效分子数，最终有效入血分子数，高危肝毒分子数量，毒性成分体内综合富集率
    return (df, com_num, f"{absorbed_mols_count} (占总比 {absorbed_mols_count/len(df):.1%})",TOXIC_THRESHOLD,
            high_toxic_count,enrichment_ratio,risk_level,advise)


    # # 导出包含你定义的 6 列字段的全景 CSV 文件
    # output_filename = csv_path.replace('.csv', '_MultiModel_Results.csv')
    # df.to_csv(output_filename, index=False)
    # print(f"\n✅ 完整 6 项预测数据已保存至: {output_filename}")
# %%
# =======================================================
# 5. 运行示例 (取消下方注释即可运行)
# =======================================================
if __name__ == '__main__':
    evaluate_tcm_formula('L6810-TCM-2928.csv', smiles_col='Smiles')
