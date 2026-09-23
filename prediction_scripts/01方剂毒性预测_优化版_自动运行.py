# -*- coding: utf-8 -*-
# 本文件由 01方剂毒性预测_优化版(3).ipynb 自动转换生成。
# 代码单元内容按 notebook 原顺序保留，未修改核心代码。

# %% [markdown]
# # 方剂毒性预测流程
#
# 本 notebook 在前一版基础上进一步减少输出文件数量和中间变量数量，核心原则如下：
#
# 1. **不再保存 Excel、JSON、TXT**，主要结果统一保存为 CSV；
# 2. **不保存参考库预览、中间去重表、重复统计表**；
# 3. 将可以合并的结果整合为少量主表；
# 4. 对统计结果统一合并到一个 `11_consolidated_count_summary.csv`；
# 5. 将上游预测阶段原来的 4 个文件进一步压缩为 3 个文件：
#    - 保留 `01_formula_compound_master.csv`
#    - 保留 `02_intoblood_prediction_results.csv`
#    - 取消 `03_formula_intoblood_for_livertox.csv`
#    - 将原 `04_livertox_prediction_results.csv` 改为 `03_livertox_prediction_results.csv`
#
# 建议保留的核心输出文件：
#
# | 文件 | 内容 |
# |---|---|
# | `01_formula_compound_master.csv` | 方剂-中药-化合物-CID-SMILES-化学分类整合主表 |
# | `02_intoblood_prediction_results.csv` | 入血预测结果，已合并化合物来源信息 |
# | `03_livertox_prediction_results.csv` | 入血成分肝毒性预测结果，已合并方剂、药材、化合物来源信息 |
# | `05_formula_toxicity_summary.csv` | 方剂层面的综合毒性评估摘要 |
# | `06_high_toxic_chemicals_master.csv` | 阳性候选成分主表，含化学分类信息 |
# | `07_high_toxic_chemical_targets.csv` | 候选成分-靶标关系 |
# | `08_high_toxic_target_pathways.csv` | 候选关联靶标-通路关系 |
# | `09_high_toxic_target_GO_terms.csv` | 候选关联靶标-GO 关系 |
# | `10_high_toxic_target_disease_results.csv` | 候选关联靶标-疾病关系 |
# | `11_consolidated_count_summary.csv` | 化学类别、靶标、通路、GO、疾病等统计结果整合表 |
# | `00_output_file_manifest.csv` | 本次输出文件清单 |
#
# > 说明：本文件需要在你本地已配置 `tcm_datasets`、`intoblood_pred`、`formula_livertox_pred`、RDKit 等依赖的环境中运行。

# %%
# ============================================================
# 0. 基础配置
# ============================================================
# 这里统一设置方剂名称、输入文件路径、输出目录、分块大小和毒性阈值。
# 后续如需更换方剂或数据路径，优先修改本单元即可。

from pathlib import Path
from datetime import datetime
import json
import os
import re
import gc

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)


def _json_string_list_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    values = json.loads(raw)
    if not isinstance(values, list):
        raise ValueError(f"{name} 必须是 JSON 数组")
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return cleaned or default

# -----------------------------
# 0.1 方剂名称设置
# -----------------------------
# 支持同时输入多个方剂名称，例如：FORMULA_LIST = ["生血宝合剂", "某某方"]
FORMULA_LIST = _json_string_list_env("TOXHERB_FORMULA_LIST_JSON", ["生血宝合剂"])

# 将方剂列表合并成一个字符串，用于输出目录命名
FORMULA_NAME = "_".join(FORMULA_LIST)

# -----------------------------
# 0.2 入血成分参考库路径
# -----------------------------
# 该路径沿用你原始 notebook 中的本地路径。
# 如果更换电脑或文件目录，请修改此路径。
RESOURCE_ROOT = Path(os.getenv("TOXHERB_RESOURCE_ROOT", Path(__file__).resolve().parents[1]))
INTOBLOOD_REF_CSV = Path(os.getenv("TOXHERB_INTOBLOOD_REF_CSV", str(RESOURCE_ROOT / "data" / "12_intoblood_ref.csv")))

# -----------------------------
# 0.3 核心字段名称
# -----------------------------
# 肝毒性预测函数默认读取 Smiles 列，因此这里统一使用 Smiles 作为列名。
SMILES_COL = "Smiles"

# -----------------------------
# 0.4 预测参数
# -----------------------------
# 每批处理的化合物数量。数据很大或内存不足时可调小，例如 300 或 500。
CHUNK_SIZE = 1000

# 是否保存每个分块的临时预测结果。
# 为了减少输出文件数量，默认不保存；只有排查中断位置时才建议改成 True。
SAVE_PART_FILES = False

# 阳性候选成分筛选阈值。
# 如果 evaluate_tcm_formula 返回了模型内部阈值，后续会优先使用模型返回阈值。
DEFAULT_CANDIDATE_SCORE_THRESHOLD = 1

# -----------------------------
# 0.5 输出目录
# -----------------------------
def safe_filename(text: str) -> str:
    """将中文/英文方剂名转换为适合文件名使用的字符串。"""
    text = str(text).strip()
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text or "formula"

RUN_ID = os.getenv("TOXHERB_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_BASE_DIR = Path(os.getenv("TOXHERB_OUTPUT_BASE_DIR", "toxicity_prediction_outputs"))
OUT_DIR = OUTPUT_BASE_DIR / f"{safe_filename(FORMULA_NAME)}_{RUN_ID}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 用于记录本次保存过的文件，最后生成 manifest。
SAVED_FILES = []

# 设置 pandas 显示参数，便于 notebook 内查看表格。
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 200)

print(f"当前分析方剂：{FORMULA_NAME}")
print(f"结果输出目录：{OUT_DIR.resolve()}")
print(f"TOXHERB_OUTPUT_DIR={OUT_DIR.resolve()}")


def toxherb_progress(stage: str, percent: int, message: str) -> None:
    print(f"TOXHERB_PROGRESS={stage}|{int(percent)}|{message}", flush=True)


toxherb_progress("entity_resolution", 8, "方剂实体解析与数据表检查")

# %%
# ============================================================
# 1. 通用工具函数
# ============================================================
# 这些函数负责字段检查、CSV 保存、频数统计和输出文件清单生成。
# 本精简版不再提供 JSON / TXT / Excel 保存函数，避免输出文件过多。

def require_columns(df: pd.DataFrame, required_cols, df_name: str = "DataFrame") -> None:
    """
    检查 DataFrame 是否包含指定字段。

    参数
    ----
    df : pd.DataFrame
        需要检查的数据表。
    required_cols : list[str]
        必须存在的列名。
    df_name : str
        数据表名称，用于报错提示。
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{df_name} 缺少必要字段：{missing_cols}\n"
            f"当前字段包括：{list(df.columns)}"
        )


def save_csv(df: pd.DataFrame, filename: str, index: bool = False) -> Path:
    """
    将 DataFrame 保存为 CSV 文件，并记录到 SAVED_FILES。

    说明
    ----
    - 统一使用 utf-8-sig 编码，便于 Windows / Excel 查看中文；
    - 本精简版只主动保存 CSV，避免 Excel 行数限制和多格式重复输出；
    - 返回完整路径，便于后续函数继续调用。
    """
    if not filename.lower().endswith(".csv"):
        filename = Path(filename).with_suffix(".csv").name

    out_path = OUT_DIR / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=index, encoding="utf-8-sig")

    # 记录文件路径、行数和列数；如果同名文件重复保存，则更新记录。
    SAVED_FILES[:] = [record for record in SAVED_FILES if record["Path"] != out_path]
    SAVED_FILES.append({
        "Path": out_path,
        "Rows": len(df),
        "Columns": df.shape[1],
    })

    print(f"已保存：{out_path}")
    return out_path


def show_table(df: pd.DataFrame, title: str = "DataFrame", max_rows: int = 10) -> None:
    """在 notebook 中显示数据表的基本形状和前若干行，不额外保存文件。"""
    print(f"\n{title}：共 {len(df)} 行，{df.shape[1]} 列")
    display(df.head(max_rows))


def show_and_save_csv(df: pd.DataFrame, filename: str, max_rows: int = 10) -> Path:
    """显示数据表前若干行，并保存为 CSV。"""
    show_table(df, filename, max_rows=max_rows)
    return save_csv(df, filename)


def value_counts_table(df, col, count_type, top_n=None):
    return count_table(df, col, count_type, top_n)


def first_existing_col(df: pd.DataFrame, candidate_cols: list[str]) -> str | None:
    """从候选字段中返回第一个实际存在于 DataFrame 中的字段名。"""
    return next((col for col in candidate_cols if col in df.columns), None)


def compact_metadata_cols(df: pd.DataFrame, preferred_cols: list[str]) -> list[str]:
    """根据实际存在字段，筛选需要保留的元数据列，避免硬编码字段不存在时报错。"""
    return [col for col in preferred_cols if col in df.columns]


def make_output_manifest() -> pd.DataFrame:
    """根据 SAVED_FILES 生成本次输出文件清单。"""
    rows = []
    for record in SAVED_FILES:
        path = record["Path"]
        if path.exists():
            rows.append({
                "FileName": path.name,
                "RelativePath": str(path.relative_to(OUT_DIR)),
                "FileType": path.suffix.lower().lstrip("."),
                "Rows": record.get("Rows"),
                "Columns": record.get("Columns"),
                "Size_MB": round(path.stat().st_size / 1024 / 1024, 4),
            })
    rows = sorted(rows, key=lambda x: x["FileName"])
    return pd.DataFrame(rows)

print("通用工具函数加载完成。")

# %%
# ============================================================
# 2. 导入数据库和预测函数
# ============================================================
# 本单元依赖你本地已有的 tcm_datasets.py、intoblood_pred.py 和 formula_livertox_pred.py。
# 如果导入失败，请先确认这些文件位于当前工作目录或 Python 可检索路径中。

from tcm_datasets import *

from rdkit import RDLogger
from intoblood_pred import match_intoblood_reference
from formula_livertox_pred import build_summary_annotations, evaluate_tcm_formula, compound_keys, count_table, candidate_target_links

# 关闭 RDKit 的部分 warning，避免大量无关提示干扰 notebook 阅读。
RDLogger.DisableLog("rdApp.warning")

print("数据库与预测函数导入完成。")

# %%
# ============================================================
# 3. 检查关键数据表字段
# ============================================================
# 目的：提前发现字段名不匹配问题，避免运行到中后期才报错。
# 如果这里报错，优先检查 tcm_datasets 中各 data_xx 表格的列名。

require_columns(data_01, ["Formula.Chinese.name", "Herb.Chinese.name"], "data_01 方剂-中药表")
require_columns(data_02, ["Herb.Chinese.name", "CID"], "data_02 中药-化合物表")
require_columns(data_03, ["CID", "SMILES"], "data_03 化合物结构表")
require_columns(data_04, ["ChemicalName", "Symbol"], "data_04 化合物-靶标表")
require_columns(data_05, ["Symbol"], "data_05 靶标-通路表")
require_columns(data_06, ["Symbol", "GOID"], "data_06 靶标-GO 表")
require_columns(data_07, ["GOID"], "data_07 GO 注释表")
require_columns(data_11, ["GeneSymbol", "GeneID", "DiseaseName", "DiseaseID"], "data_11 靶标-疾病表")

if not INTOBLOOD_REF_CSV.exists():
    raise FileNotFoundError(
        f"未找到入血成分参考库：{INTOBLOOD_REF_CSV}\n"
        "请在第 0 单元修改 INTOBLOOD_REF_CSV 为你的真实文件路径。"
    )

print("关键字段检查通过。")

# %% [markdown]
# ## 4. 方剂 → 中药 → 化合物 → SMILES 整合主表
#
# 这一节不再分别保存“方剂中药组成”“中药成分表”“SMILES 全表”“去重预测表”，而是统一整合为一个主表：
#
# `01_formula_compound_master.csv`
#
# 其中会额外增加两个标记字段：
#
# - `Has_SMILES`：是否匹配到有效 SMILES；
# - `For_Intoblood_Prediction`：是否作为唯一 SMILES 输入入血预测模型。

# %%
# ============================================================
# 4.1 构建方剂-中药-化合物-SMILES 整合主表
# ============================================================

# 1）筛选方剂对应的中药组成
herb_df = (
    data_01
    .loc[data_01["Formula.Chinese.name"].isin(FORMULA_LIST)]
    .drop_duplicates()
    .reset_index(drop=True)
)

if herb_df.empty:
    raise ValueError(
        f"没有在 data_01 中检索到方剂：{FORMULA_LIST}\n"
        "请检查方剂中文名称是否与数据库完全一致。"
    )

# 2）将结构表中的 SMILES 字段统一重命名为 Smiles，便于后续模型调用
structure_df = data_03.rename(columns={"SMILES": SMILES_COL}).copy()

# 3）从方剂中药组成出发，依次合并中药-化合物表和化合物结构表
compound_master_df = (
    herb_df[["Formula.Chinese.name", "Herb.Chinese.name"]]
    .drop_duplicates()
    .merge(data_02, on="Herb.Chinese.name", how="left")
    .merge(structure_df, on="CID", how="left", suffixes=("", "_structure"))
    .drop_duplicates()
    .reset_index(drop=True)
)

# 4）增加是否匹配到 SMILES 的标记
compound_master_df["Has_SMILES"] = compound_master_df[SMILES_COL].notna()
compound_master_df["Compound_Key"] = compound_keys(compound_master_df)
if "Herb.Chinese.name" in compound_master_df:
    source_herbs = compound_master_df.groupby("Compound_Key")["Herb.Chinese.name"].agg(lambda values: ";".join(sorted(set(values.dropna().astype(str)))))
    compound_master_df["Source_Herbs"] = compound_master_df["Compound_Key"].map(source_herbs)
if "Formula.Chinese.name" in compound_master_df:
    source_formulas = compound_master_df.groupby("Compound_Key")["Formula.Chinese.name"].agg(lambda values: ";".join(sorted(set(values.dropna().astype(str)))))
    compound_master_df["Source_Formulas"] = compound_master_df["Compound_Key"].map(source_formulas)


# 5）同一 SMILES 只保留第一条进入入血预测，避免同一分子重复预测
compound_master_df["For_Intoblood_Prediction"] = False
valid_smiles_index = (
    compound_master_df
    .drop_duplicates(subset=["Compound_Key"], keep="first")
    .index
)
compound_master_df.loc[valid_smiles_index, "For_Intoblood_Prediction"] = True

# 6）实际用于入血预测的唯一 SMILES 输入表；该表仅作为内存变量，不单独写出
prediction_input_df = (
    compound_master_df
    .loc[compound_master_df["For_Intoblood_Prediction"]]
    .copy()
    .reset_index(drop=True)
)

if prediction_input_df.empty:
    raise ValueError("匹配 SMILES 后没有可用于预测的化合物，请检查 data_03 的 CID 与 SMILES。")

# 7）只保存一个整合主表，不再保存多个中间表
show_and_save_csv(compound_master_df, "01_formula_compound_master.csv")
toxherb_progress("entity_resolution", 20, "方剂-中药-成分映射完成")

print(f"方剂中药数量：{herb_df['Herb.Chinese.name'].nunique()}")
print(f"方剂原始化合物记录数：{len(compound_master_df)}")
print(f"匹配到 SMILES 的记录数：{compound_master_df['Has_SMILES'].sum()}")
print(f"用于入血筛选的唯一成分数：{len(prediction_input_df)}")

# %% [markdown]
# ## 5. 入血成分预测
#
# 这一节不再保存入血参考库预览，只读取参考库并直接进行预测。
#
# 本节只正式输出 1 个 CSV：
#
# - `02_intoblood_prediction_results.csv`：全部候选化合物的入血预测结果，已合并方剂、药材、化合物来源信息。
#
# 说明：
#
# - 原来的 `03_formula_intoblood_for_livertox.csv` 不再作为正式输出文件保存；
# - 预测为入血的成分仅保存在内存变量 `intoblood_for_livertox_df` 中；
# - 因为 `evaluate_tcm_formula()` 需要读取 CSV 路径，后续会临时生成一个 `_temporary_livertox_input.csv`，预测完成后自动删除。

# %%
# ============================================================
# 5.1 读取入血成分参考库
# ============================================================
# 为减少输出文件数量，这里只读取参考库，不再保存 reference preview。

intoblood_reference_df = pd.read_csv(INTOBLOOD_REF_CSV).rename(columns={"SMILES": SMILES_COL})
require_columns(
    intoblood_reference_df,
    ["CID", "ChemicalName", SMILES_COL, "IntoBlood"],
    "入血成分参考库 intoblood_reference_df",
)

print(f"入血成分参考库读取完成：{len(intoblood_reference_df)} 行，{intoblood_reference_df.shape[1]} 列")

# %%
# ============================================================
# 5.2 分块进行入血成分预测
# ============================================================

def predict_intoblood_in_chunks(
    query_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    query_smiles_col: str = "Smiles",
    ref_smiles_col: str = "Smiles",
    chunk_size: int = 1000,
    save_part_files: bool = False,
) -> pd.DataFrame:
    """
    分块进行入血成分预测，降低大数据量时的内存压力。

    参数
    ----
    query_df : pd.DataFrame
        待预测化合物表，必须包含 query_smiles_col。
    reference_df : pd.DataFrame
        入血成分参考库，必须包含 ref_smiles_col。
    query_smiles_col : str
        待预测表中的 SMILES 列名。
    ref_smiles_col : str
        参考库中的 SMILES 列名。
    chunk_size : int
        每批预测数量。
    save_part_files : bool
        是否保存每批临时结果。为了减少输出文件，默认 False。

    返回
    ----
    pd.DataFrame
        合并后的全部入血预测结果。
    """
    require_columns(query_df, [query_smiles_col], "query_df")
    require_columns(reference_df, [ref_smiles_col], "reference_df")

    n_total = len(query_df)
    if n_total == 0:
        return pd.DataFrame()

    part_results = []
    part_dir = OUT_DIR / "intoblood_part_files"
    if save_part_files:
        part_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, start in enumerate(range(0, n_total, chunk_size), start=1):
        end = min(start + chunk_size, n_total)
        print(f"正在处理第 {batch_idx} 批：第 {start + 1} - {end} 行 / 共 {n_total} 行")

        # 复制当前分块，避免函数内部修改原始数据
        chunk = query_df.iloc[start:end].copy()

        # 按 CID、ChemicalName、Smiles 依次精确匹配入血参考库
        pred = match_intoblood_reference(
            query_df=chunk,
            reference_df=reference_df,
            verbose=True,
        )

        # 兼容不同版本函数返回字段，只保留存在的核心预测字段
        core_cols = [
            "Compound_Key",
            "Precomputed_Match",
            "Bioavailability_Ma",
            "Reference_Match",
            "IntoBlood",
            "IntoBlood_Reason",
        ]
        keep_cols = [col for col in core_cols if col in pred.columns]
        pred_core = pred.loc[:, keep_cols].copy()
        pred_core["Batch_ID"] = batch_idx
        part_results.append(pred_core)

        # 可选保存分块结果；默认关闭，避免产生大量 part 文件
        if save_part_files:
            part_file = part_dir / f"intoblood_part_{batch_idx:03d}.csv"
            pred_core.to_csv(part_file, index=False, encoding="utf-8-sig")
            SAVED_FILES.append(part_file)
            print(f"已保存分块文件：{part_file}")

        # 主动释放大对象，降低内存占用
        del chunk, pred, pred_core
        gc.collect()

    return pd.concat(part_results, ignore_index=True)


# 1）执行入血成分预测，只对唯一 SMILES 进行预测
intoblood_pred_core_df = predict_intoblood_in_chunks(
    query_df=prediction_input_df,
    reference_df=intoblood_reference_df,
    query_smiles_col=SMILES_COL,
    ref_smiles_col=SMILES_COL,
    chunk_size=CHUNK_SIZE,
    save_part_files=SAVE_PART_FILES,
)

if intoblood_pred_core_df.empty:
    raise ValueError("入血成分预测结果为空，请检查输入化合物和参考库。")

# 2）将预测结果合并回方剂-化合物主表，便于追踪入血成分来源于哪些药材和化合物
intoblood_df = (
    prediction_input_df
    .merge(intoblood_pred_core_df, on="Compound_Key", how="left", suffixes=("", "_pred"))
    .drop_duplicates()
    .reset_index(drop=True)
)

show_and_save_csv(intoblood_df, "02_intoblood_prediction_results.csv")
toxherb_progress("intoblood_prediction", 42, "入血预测完成")

# %%
# ============================================================
# 5.3 提取预测为入血的化合物
# ============================================================
# IntoBlood == 1 表示预测为可能入血的成分。
# 注意：本表只作为下一步肝毒性预测的内存变量，不再正式保存为 CSV。
# 这样可以减少一个中间输出文件：03_formula_intoblood_for_livertox.csv。

require_columns(intoblood_df, ["IntoBlood"], "intoblood_df")

intoblood_for_livertox_df = (
    intoblood_df
    .drop_duplicates(subset=["Compound_Key"])
    .reset_index(drop=True)
)

show_table(intoblood_for_livertox_df, "入血成分内存表：intoblood_for_livertox_df")
print(f"入血规则判定通过的唯一化合物数量：{int((intoblood_for_livertox_df['IntoBlood'] == 1).sum())}")

# %% [markdown]
# ## 6. 入血成分的多层级肝毒性预测
#
# 调用你已有的 `evaluate_tcm_formula()` 函数，对入血成分进行细胞、动物、临床层面的肝毒性预测。
#
# 本节输出两个 CSV：
#
# - `03_livertox_prediction_results.csv`：每个入血成分的肝毒性预测结果；
# - `05_formula_toxicity_summary.csv`：方剂层面的综合毒性评估摘要。
#
# 说明：
#
# - 原来的 `04_livertox_prediction_results.csv` 改名为 `03_livertox_prediction_results.csv`；
# - 原来的 `03_formula_intoblood_for_livertox.csv` 不再正式保存，只临时写出供模型函数读取，预测完成后自动删除；
# - 后续 `05`、`06`、`07`、`08`、`09`、`10`、`11` 等输出文件保持不变。

# %%
# ============================================================
# 6.1 运行方剂肝毒性综合评估
# ============================================================
# evaluate_tcm_formula() 是你已有的预测函数。
# 该函数目前需要读取 CSV 文件路径，因此这里临时写出一个输入 CSV。
# 预测完成后会自动删除该临时文件，不作为正式输出文件保留。

temp_livertox_input_path = OUT_DIR / "_temporary_livertox_input.csv"

try:
    # 1）临时保存入血成分输入表，供 evaluate_tcm_formula() 读取
    intoblood_for_livertox_df.to_csv(
        temp_livertox_input_path,
        index=False,
        encoding="utf-8-sig",
    )

    # 2）调用肝毒性综合预测函数
    (
        livertox_result_raw_df,
        original_valid_molecule_count,
        absorbed_molecule_count_or_percent,
        model_toxic_threshold,
        high_toxic_molecule_count,
        toxic_enrichment_ratio,
        risk_level,
        advice,
    ) = evaluate_tcm_formula(
        str(temp_livertox_input_path),
        smiles_col=SMILES_COL,
    )
finally:
    # 3）无论预测是否成功，都尽量删除临时输入文件，保持输出目录简洁
    if temp_livertox_input_path.exists():
        temp_livertox_input_path.unlink()

# 如果函数返回了阈值，则优先使用函数返回阈值；否则使用默认阈值
CANDIDATE_SCORE_THRESHOLD = (
    model_toxic_threshold
    if model_toxic_threshold is not None
    else DEFAULT_CANDIDATE_SCORE_THRESHOLD
)

# 4）将肝毒性结果与入血成分来源信息合并，减少后续回溯麻烦
livertox_df = livertox_result_raw_df.copy()
# 5）正式保存肝毒性预测结果。
# 原来的 04_livertox_prediction_results.csv 改为 03_livertox_prediction_results.csv。
show_and_save_csv(livertox_df, "03_livertox_prediction_results.csv")
toxherb_progress("livertox_prediction", 62, "肝毒性多模型预测完成")

print(f"肝毒性阈值：{CANDIDATE_SCORE_THRESHOLD}")
summary_annotations = build_summary_annotations(
    livertox_df,
    herb_candidates=herb_df["Herb.Chinese.name"].dropna().unique().tolist(),
    absorbed_molecule_count_or_percent=absorbed_molecule_count_or_percent,
)

# %%
# ============================================================
# 6.2 生成并保存方剂层面的综合评估摘要
# ============================================================
# 本精简版只保存 CSV，不再额外保存 JSON 和 TXT。

summary_df = pd.DataFrame([{
    "Formula": FORMULA_NAME,
    "Original_valid_molecule_count": original_valid_molecule_count,
    "Absorbed_molecule_count_or_percent": absorbed_molecule_count_or_percent,
    "Candidate_score_threshold": CANDIDATE_SCORE_THRESHOLD,
    "High_toxic_molecule_count": high_toxic_molecule_count,
    "Candidate_ratio": toxic_enrichment_ratio,
    "Screening_status": risk_level,
    "Advice": advice,
    **summary_annotations,
    "Output_dir": str(OUT_DIR.resolve()),
}])

show_and_save_csv(summary_df, "05_formula_toxicity_summary.csv")

print("\n" + "=" * 60)
print("📊 方剂肝毒性系统生物学评估报告")
print("=" * 60)
print(f"▶ 方剂名称: {FORMULA_NAME}")
print(f"▶ 原始检出有效分子数: {original_valid_molecule_count}")
print(f"▶ 最终入血候选成分数/比例: {absorbed_molecule_count_or_percent}")
print(f"▶ 阳性候选成分数 (阳性端点数 >= {CANDIDATE_SCORE_THRESHOLD}): {high_toxic_molecule_count}")
print(f"▶ 阳性候选成分比例: {toxic_enrichment_ratio:.2%}" if isinstance(toxic_enrichment_ratio, (int, float)) else f"▶ 阳性候选成分比例: {toxic_enrichment_ratio}")
print(f"▶ 整体评价: {risk_level}")
print(f"▶ 综合建议: {advice}")
print(f"▶ 药食同源标注: {summary_annotations['Food_medicine_homology_note']}")
print("=" * 60)

# %% [markdown]
# ## 7. 阳性候选成分筛选
#
# 这一节不再分别保存：
#
# - `10_high_toxic_chemicals.csv`
# - `11_high_toxic_chemicals_with_classification.csv`
# - `12_high_toxic_class_counts.csv`
# - `13_high_toxic_superclass_counts.csv`
# - `14_high_toxic_pathway_class_counts.csv`
# - `15_high_toxic_glycoside_counts.csv`
#
# 而是整合为：
#
# - `06_high_toxic_chemicals_master.csv`：候选成分主表；
# - 后续统一统计结果进入 `11_consolidated_count_summary.csv`。

# %%
# ============================================================
# 7.1 筛选阳性候选成分并合并化学分类信息
# ============================================================

require_columns(livertox_df, ["Max_Tox_Prob", "Hepatotoxicity_Score"], "livertox_df")

high_toxic_df = (
    livertox_df
    .loc[livertox_df["Hepatotoxicity_Score"] >= CANDIDATE_SCORE_THRESHOLD]
    .drop_duplicates(subset=["Compound_Key"])
    .sort_values("Max_Tox_Prob", ascending=False)
    .reset_index(drop=True)
)

show_and_save_csv(high_toxic_df, "06_high_toxic_chemicals_master.csv")

print(f"阳性候选成分数量：{len(high_toxic_df)}")

# %% [markdown]
# ## 8. 候选成分靶标、通路、GO 和疾病关联分析
#
# 这一部分仍然保留 4 个关系表，因为它们代表不同层级的数据关系，不建议强行合并成一个宽表，否则会因为通路、GO、疾病之间的多对多关系导致结果膨胀和解释混乱。

# %%
# ============================================================
# 8.1 候选成分关联靶标：保留候选 CID 与来源，不按共享结构扩张集合。
# ============================================================
target_df = candidate_target_links(high_toxic_df, data_02, data_04)

show_and_save_csv(target_df, "07_high_toxic_chemical_targets.csv")

print(f"候选成分关联靶标数量：{target_df['Symbol'].nunique() if not target_df.empty else 0}")

# %%
# ============================================================
# 8.2 高危毒性成分关联通路
# ============================================================
# 输出：08_high_toxic_target_pathways.csv

pathway_df = (
    target_df[["Symbol"]]
    .drop_duplicates()
    .merge(data_05, on="Symbol", how="left")
    .drop_duplicates()
    .reset_index(drop=True)
)

show_and_save_csv(pathway_df, "08_high_toxic_target_pathways.csv")

candidate_pathway_cols = ["Pathway", "PathwayName", "Pathway.Name", "Term", "Description", "Name"]
pathway_name_col = first_existing_col(pathway_df, candidate_pathway_cols)
print(f"用于通路统计的字段：{pathway_name_col}")

# %%
# ============================================================
# 8.3 高危毒性成分关联 GO 术语
# ============================================================
# 输出：09_high_toxic_target_GO_terms.csv

go_df = (
    target_df[["Symbol"]]
    .drop_duplicates()
    .merge(data_06, on="Symbol", how="left")
    .merge(data_07, on="GOID", how="left")
    .drop_duplicates()
    .reset_index(drop=True)
)

show_and_save_csv(go_df, "09_high_toxic_target_GO_terms.csv")

candidate_go_cols = ["GOTerm", "GO.Term", "Term", "Name", "Description"]
go_name_col = first_existing_col(go_df, candidate_go_cols)
print(f"用于 GO 统计的字段：{go_name_col}")

# %%
# ============================================================
# 8.4 高危毒性成分可能关联疾病
# ============================================================
# 输出：10_high_toxic_target_disease_results.csv
# 注意：该结果可能非常大，但疾病关联表本身是一个核心结果，因此仍然保留为独立 CSV。

disease_df = (
    target_df[["Symbol"]]
    .drop_duplicates()
    .merge(data_11, left_on="Symbol", right_on="GeneSymbol", how="left")
    .loc[:, ["Symbol", "GeneID", "DiseaseName", "DiseaseID"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

show_and_save_csv(disease_df, "10_high_toxic_target_disease_results.csv")
toxherb_progress("mechanism_mapping", 84, "靶标、通路、GO与疾病映射完成")

print(f"疾病关联记录数：{len(disease_df)}")
print(f"唯一疾病数量：{disease_df['DiseaseName'].nunique() if 'DiseaseName' in disease_df.columns else 0}")

# %% [markdown]
# ## 9. 统一统计汇总表
#
# 这一节将原来分散保存的多个统计表整合为一个长表：
#
# `11_consolidated_count_summary.csv`
#
# 该表用 `Count_Type` 区分统计类型，便于筛选、排序和后续绘图。

# %%
# ============================================================
# 9.1 构建统一统计汇总表
# ============================================================
# 将候选成分化学类别、靶标、通路、GO、疾病统计结果整合到一个 CSV。

count_tables = []

# 1）候选成分化学分类统计
count_tables.append(value_counts_table(high_toxic_df, "Class", "High_toxic_chemical_class"))
count_tables.append(value_counts_table(high_toxic_df, "Superclass", "High_toxic_chemical_superclass"))
count_tables.append(value_counts_table(high_toxic_df, "Pathway", "High_toxic_chemical_pathway_class"))
count_tables.append(value_counts_table(high_toxic_df, "Is_glycoside", "High_toxic_glycoside"))

# 2）候选成分关联靶标统计
count_tables.append(value_counts_table(target_df, "Symbol", "High_toxic_target"))

# 3）候选关联靶标通路统计
if pathway_name_col is not None:
    count_tables.append(value_counts_table(pathway_df, pathway_name_col, "High_toxic_target_pathway"))

# 4）候选关联靶标 GO 统计
if go_name_col is not None:
    count_tables.append(value_counts_table(go_df, go_name_col, "High_toxic_target_GO"))

# 5）候选关联靶标疾病统计
count_tables.append(value_counts_table(disease_df, "DiseaseName", "High_toxic_target_disease"))

# 合并所有非空统计表
count_tables = [df for df in count_tables if not df.empty]
if count_tables:
    count_summary_df = pd.concat(count_tables, ignore_index=True)
else:
    count_summary_df = pd.DataFrame(columns=["Count_Type", "Source_Column", "Item", "Count", "Percent"])

show_and_save_csv(count_summary_df, "11_consolidated_count_summary.csv")

# %% [markdown]
# ## 10. 输出文件清单
#
# 最后自动生成本次运行保存的
# CSV
# 文件清单，方便检查输出是否完整。
#

# %%
# ============================================================
# 10.1 生成输出文件清单
# ============================================================

# 第一次生成并保存 manifest
manifest_df = make_output_manifest()
save_csv(manifest_df, "00_output_file_manifest.csv")

# 第二次重新生成，让 manifest 自身也记录进清单
manifest_df = make_output_manifest()
save_csv(manifest_df, "00_output_file_manifest.csv")

show_table(manifest_df, "本次输出文件清单", max_rows=50)

print(f"全部结果已保存到：{OUT_DIR.resolve()}")
toxherb_progress("complete", 100, "方剂预测报告生成完成")
