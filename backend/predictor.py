from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import queue
import re
import runpy
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED

from prediction_scripts.intoblood_pred import match_intoblood_reference
from prediction_scripts.model_runtime import (
    load_model_bundles,
    predict_multimodel_pipeline as predict_model_bundles,
)

try:
    import torch
except ImportError:
    torch = None


RDLogger.DisableLog("rdApp.*")


def is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def get_resource_root(default_root: Path | None = None) -> Path:
    override = os.getenv("TOXHERB_RESOURCE_ROOT")
    if override:
        return Path(override).resolve()
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root).resolve()
    if default_root is not None:
        return Path(default_root).resolve()
    return Path(__file__).resolve().parents[1]


def get_user_dir() -> Path:
    override = os.getenv("TOXHERB_USER_DIR")
    if override:
        root = Path(override)
    elif os.name == "nt" and os.getenv("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"]) / "ToxHERB"
    else:
        root = Path.home() / ".toxherb"
    root.mkdir(parents=True, exist_ok=True)
    return root


class PredictionError(RuntimeError):
    """Domain-level error that can be safely shown through the API."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class SystemPaths:
    root: Path
    model_cell: Path
    model_animal: Path
    model_clinical: Path
    formula_herb: Path
    herb_compound: Path
    compound_class: Path
    compound_target: Path
    target_pathway: Path
    target_go: Path
    target_disease: Path
    chem_diseases: Path
    chem_go: Path
    chem_pathways: Path
    go_term: Path
    intoblood_reference: Path
    food_medicine_homology: Path
    ctd_cache: Path
    ctd_cache_seed: Path

    @classmethod
    def from_root(cls, root: Path) -> "SystemPaths":
        bundled_ctd_cache = root / "data" / "00_ctd_lookup.sqlite"
        ctd_cache = bundled_ctd_cache
        if is_frozen_app():
            ctd_cache = get_user_dir() / "ctd_cache" / "00_ctd_lookup.sqlite"
        return cls(
            root=root,
            model_cell=root / "models" / "01best_model_tuned.joblib",
            model_animal=root / "models" / "02best_model_tuned.joblib",
            model_clinical=root / "models" / "03best_model_tuned.joblib",
            formula_herb=root / "data" / "01_formula_herb.csv",
            herb_compound=root / "data" / "02_herb_compound.csv",
            compound_class=root / "data" / "03_compound_class.csv",
            compound_target=root / "data" / "04_compound_target.csv",
            target_pathway=root / "data" / "05_target_pathway.csv",
            target_go=root / "data" / "06_target_go.csv",
            go_term=root / "data" / "07_go_term.csv",
            target_disease=root / "data" / "08_target_disease.csv",
            chem_diseases=root / "data" / "09_chem_diseases.csv",
            chem_go=root / "data" / "10_chem_go.csv",
            chem_pathways=root / "data" / "11_chem_pathways.csv",
            intoblood_reference=root / "data" / "12_intoblood_ref.csv",
            food_medicine_homology=root / "data" / "13_food_medicine_homology.csv",
            ctd_cache=ctd_cache,
            ctd_cache_seed=bundled_ctd_cache,
        )


def split_items(text: str | Iterable[str]) -> list[str]:
    if isinstance(text, str):
        pieces = re.split(r"[\n,，;；、]+", text)
    else:
        pieces = []
        for item in text:
            pieces.extend(re.split(r"[\n,，;；、]+", str(item)))
    return [piece.strip() for piece in pieces if piece and piece.strip()]


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise PredictionError(f"缺少数据文件: {path}", status_code=500)
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk", **kwargs)


CSV_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "SMILES": ("Smiles", "smiles"),
    "Smiles": ("SMILES", "smiles"),
    "TERM": ("GOTerm", "GOTermName", "term"),
    "GOTerm": ("TERM", "GOTermName", "term"),
    "GOTermName": ("GOTerm", "TERM", "term"),
    "GOID": ("GOTermID", "goid"),
    "GOTermID": ("GOID", "goid"),
    "HerbName": ("Herb.Chinese.name", "Herb", "ChineseName"),
    "Symbol": ("GeneSymbol", "genesymbol", "Target"),
    "GeneSymbol": ("Symbol", "genesymbol", "Target"),
    "ENTREZID": ("GeneID", "EntrezID", "entrezid"),
    "GeneID": ("ENTREZID", "EntrezID", "entrezid"),
    "Pathwayid": ("PathwayID", "pathwayid"),
    "PathwayID": ("Pathwayid", "pathwayid"),
}

OUTPUT_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "GOID": ("GOTermID", "GOTermID_x", "GOTermID_y", "GOID_x", "GOID_y", "goid"),
    "GOTermID": ("GOID", "GOID_x", "GOID_y", "GOTermID_x", "GOTermID_y", "goid"),
    "TERM": ("GOTerm", "GOTermName", "TERM_x", "TERM_y", "GOTerm_x", "GOTerm_y", "GOTermName_x", "GOTermName_y", "term"),
    "GOTerm": ("TERM", "GOTermName", "TERM_x", "TERM_y", "GOTerm_x", "GOTerm_y", "GOTermName_x", "GOTermName_y", "term"),
    "GOTermName": ("GOTerm", "TERM", "TERM_x", "TERM_y", "GOTerm_x", "GOTerm_y", "GOTermName_x", "GOTermName_y", "term"),
    "Symbol": ("GeneSymbol", "GeneSymbol_x", "GeneSymbol_y", "GeneSymbol_disease", "Symbol_x", "Symbol_y", "Symbol_disease", "genesymbol", "Target"),
    "GeneSymbol": ("Symbol", "Symbol_x", "Symbol_y", "Symbol_disease", "GeneSymbol_x", "GeneSymbol_y", "GeneSymbol_disease", "genesymbol", "Target"),
    "ENTREZID": ("GeneID", "GeneID_x", "GeneID_y", "GeneID_disease", "EntrezID", "ENTREZID_x", "ENTREZID_y", "ENTREZID_disease", "entrezid"),
    "GeneID": ("ENTREZID", "ENTREZID_x", "ENTREZID_y", "ENTREZID_disease", "EntrezID", "GeneID_x", "GeneID_y", "GeneID_disease", "entrezid"),
    "Pathwayid": ("PathwayID", "Pathwayid_x", "Pathwayid_y", "PathwayID_x", "PathwayID_y", "pathwayid"),
    "PathwayID": ("Pathwayid", "Pathwayid_x", "Pathwayid_y", "PathwayID_x", "PathwayID_y", "pathwayid"),
    "PathwayName": ("PathwayName_x", "PathwayName_y", "pathway"),
    "DiseaseName": ("DiseaseName_x", "DiseaseName_y", "disease_name"),
    "DiseaseID": ("DiseaseID_x", "DiseaseID_y", "disease_id"),
}

OUTPUT_ALIAS_DROP_COLUMNS = {
    "GOTermID_x",
    "GOTermID_y",
    "GOID_x",
    "GOID_y",
    "TERM_x",
    "TERM_y",
    "GOTerm_x",
    "GOTerm_y",
    "GOTermName_x",
    "GOTermName_y",
    "GeneID",
    "GeneID_x",
    "GeneID_y",
    "GeneID_disease",
    "GeneSymbol",
    "GeneSymbol_x",
    "GeneSymbol_y",
    "GeneSymbol_disease",
    "ENTREZID_x",
    "ENTREZID_y",
    "ENTREZID_disease",
    "Symbol_x",
    "Symbol_y",
    "Symbol_disease",
    "PathwayID",
    "PathwayID_x",
    "PathwayID_y",
    "Pathwayid_x",
    "Pathwayid_y",
}

SECTION_CORE_COLUMNS: dict[str, list[str]] = {
    "toxic_go": ["Symbol", "GOID", "Ontology", "TERM"],
    "toxic_pathways": ["Symbol", "ENTREZID", "Pathwayid", "PathwayName"],
    "toxic_diseases": ["ChemicalName", "ENTREZID", "Symbol", "DiseaseName", "DiseaseID"],
}

PREDICTION_RESPONSE_SCHEMA_VERSION = "toxherb-response-20260730-reference-match-v6"


def _copy_column_aliases(
    df: pd.DataFrame,
    aliases: dict[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    alias_map = aliases or CSV_COLUMN_ALIASES
    out = df.copy()
    for canonical, candidates in alias_map.items():
        if canonical in out.columns:
            continue
        source = next((candidate for candidate in candidates if candidate in out.columns), None)
        if source is not None:
            out[canonical] = out[source]
    return out


def _blank_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.fillna("").astype(str).str.strip().isin({"", "未提供", "nan", "NaN", "None"})


def _identifier_text(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if re.fullmatch(r"\d+(\.0+)?", text):
        return str(int(float(text)))
    return text


def _coalesce_relation_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    for canonical, aliases in OUTPUT_COLUMN_ALIASES.items():
        candidates = [canonical, *aliases]
        present = [column for column in candidates if column in out.columns]
        if not present:
            continue
        merged = pd.Series(pd.NA, index=out.index, dtype=object)
        for column in present:
            values = out[column]
            mask = _blank_mask(merged) & ~_blank_mask(values)
            merged = merged.mask(mask, values)
        out[canonical] = merged
    return out


def _normalize_relation_output_frame(df: pd.DataFrame, section_id: str | None = None) -> pd.DataFrame:
    if df.empty:
        columns = SECTION_CORE_COLUMNS.get(section_id or "")
        return pd.DataFrame(columns=columns) if columns else df.copy()
    out = _coalesce_relation_columns(df)
    drop_columns = [column for column in OUTPUT_ALIAS_DROP_COLUMNS if column in out.columns]
    if drop_columns:
        out = out.drop(columns=drop_columns)
    if "ENTREZID" in out.columns:
        out["ENTREZID"] = out["ENTREZID"].map(_identifier_text)
    core_columns = SECTION_CORE_COLUMNS.get(section_id or "")
    if core_columns:
        for column in core_columns:
            if column not in out.columns:
                out[column] = pd.NA
        out = out[core_columns]
    return out


def _drop_blank_relation_rows(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    mask = pd.Series(False, index=df.index)
    for column in required_columns:
        if column in df.columns:
            mask |= ~_blank_mask(df[column])
    return df[mask].copy()


def _valid_toxic_disease_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(False, index=df.index)
    normalized = _normalize_relation_output_frame(df, "toxic_diseases")
    chemical_ok = ~_blank_mask(normalized["ChemicalName"]) if "ChemicalName" in normalized.columns else pd.Series(False, index=normalized.index)
    disease_ok = pd.Series(False, index=normalized.index)
    for column in ("DiseaseName", "DiseaseID"):
        if column in normalized.columns:
            disease_ok |= ~_blank_mask(normalized[column])
    return chemical_ok & disease_ok


def _filter_toxic_disease_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = _normalize_relation_output_frame(df, "toxic_diseases")
    return out[_valid_toxic_disease_mask(out)].copy()


def _read_csv_header(path: Path) -> list[str]:
    return list(_read_csv(path, nrows=0).columns)


def _resolve_csv_usecols(path: Path, columns: list[str]) -> list[str]:
    header = _read_csv_header(path)
    resolved: list[str] = []
    missing: list[str] = []
    for column in columns:
        candidates = (column, *CSV_COLUMN_ALIASES.get(column, ()))
        source = next((candidate for candidate in candidates if candidate in header), None)
        if source is None:
            missing.append(column)
        elif source not in resolved:
            resolved.append(source)
    if missing:
        raise PredictionError(
            f"Data file {path.name} is missing columns: {', '.join(missing)}",
            status_code=500,
        )
    return resolved


def _read_csv_chunks_with_aliases(
    path: Path,
    columns: list[str],
    chunksize: int,
    **kwargs: Any,
) -> Iterable[pd.DataFrame]:
    usecols = _resolve_csv_usecols(path, columns)
    try:
        reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize, **kwargs)
        for chunk in reader:
            yield _copy_column_aliases(chunk)
    except UnicodeDecodeError:
        reader = pd.read_csv(path, usecols=usecols, encoding="gbk", chunksize=chunksize, **kwargs)
        for chunk in reader:
            yield _copy_column_aliases(chunk)


def _native(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set, dict)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    safe_df = df.head(limit).replace({np.nan: None})
    output: list[dict[str, Any]] = []
    for row in safe_df.to_dict(orient="records"):
        output.append({str(k): _native(v) for k, v in row.items()})
    return output


def _balanced_records(
    df: pd.DataFrame,
    group_columns: list[str],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if df.empty:
        return []
    safe_df = df.drop_duplicates().copy()
    if limit is None or len(safe_df) <= limit:
        return _records(safe_df, None)

    group_col = next((col for col in group_columns if col in safe_df.columns), None)
    if not group_col:
        return _records(safe_df, limit)

    keyed = safe_df.copy()
    keyed["_balance_key"] = keyed[group_col].fillna("").astype(str).str.strip()
    grouped = [
        group.drop(columns=["_balance_key"]).reset_index(drop=True)
        for _, group in keyed.groupby("_balance_key", sort=False)
        if str(_).strip()
    ]
    if not grouped:
        return _records(safe_df, limit)

    effective_limit = max(limit, len(grouped))
    selected: list[dict[str, Any]] = []
    row_index = 0
    while len(selected) < effective_limit:
        progressed = False
        for group in grouped:
            if row_index < len(group):
                selected.append(group.iloc[row_index].to_dict())
                progressed = True
                if len(selected) >= effective_limit:
                    break
        if not progressed:
            break
        row_index += 1
    return [{str(k): _native(v) for k, v in row.items()} for row in selected]


EXTERNAL_CSV_SECTION_ROW_LIMIT = 1000
EXTERNAL_RESPONSE_RECORD_LIMIT = 500
EXTERNAL_COMPOUND_RECORD_LIMIT = 1000
EXTERNAL_SECTION_CHUNK_SIZE = 100_000
EXTERNAL_LARGE_CSV_THRESHOLD_BYTES = 50 * 1024 * 1024
DISCLAIMER_TEXT = "本系统结果仅用于科研辅助和风险优先级排序，不作为处方、停药或安全性最终判断。"
HEPATIC_KEYWORDS = (
    "liver",
    "hepatic",
    "hepat",
    "cholestasis",
    "bile",
    "fibrosis",
    "steatosis",
    "cirrhosis",
    "肝",
    "胆汁",
    "胆酸",
    "炎症",
    "氧化",
    "线粒体",
    "脂质",
)


EXTERNAL_PREDICTION_CONFIG: dict[str, dict[str, Any]] = {
    "formula": {
        "script": "01方剂毒性预测_优化版_自动运行.py",
        "env_var": "TOXHERB_FORMULA_LIST_JSON",
        "summary_file": "05_formula_toxicity_summary.csv",
        "livertox_file": "03_livertox_prediction_results.csv",
        "overview_files": [
            ("输出文件清单", "00_output_file_manifest.csv"),
            ("方剂-中药-化学成分主表", "01_formula_compound_master.csv"),
            ("入血判定结果", "02_intoblood_prediction_results.csv"),
            ("肝毒性预测结果", "03_livertox_prediction_results.csv"),
            ("方剂毒性摘要", "05_formula_toxicity_summary.csv"),
            ("综合统计结果", "11_consolidated_count_summary.csv"),
        ],
        "tab_files": {
            "toxic_compounds": ("高危肝毒成分明细", "06_high_toxic_chemicals_master.csv"),
            "toxic_targets": ("毒性成分对应靶标", "07_high_toxic_chemical_targets.csv"),
            "toxic_pathways": ("毒性靶标-信号通路关系", "08_high_toxic_target_pathways.csv"),
            "toxic_go": ("毒性靶标-GO术语关系", "09_high_toxic_target_GO_terms.csv"),
            "toxic_diseases": ("毒性靶标-疾病关系", "10_high_toxic_target_disease_results.csv"),
        },
    },
    "herb": {
        "script": "02中药毒性预测_优化版_最终版_自动运行.py",
        "env_var": "TOXHERB_CHINESE_MEDICINES_JSON",
        "summary_file": "05_chinese_medicine_toxicity_summary.csv",
        "livertox_file": "04_livertox_prediction_results.csv",
        "overview_files": [
            ("输出文件清单", "00_output_file_manifest.csv"),
            ("中药相关方剂", "01_chinese_medicine_related_formulae.csv"),
            ("中药-化学成分主表", "02_chinese_medicine_compound_master.csv"),
            ("入血判定结果", "03_intoblood_prediction_results.csv"),
            ("肝毒性预测结果", "04_livertox_prediction_results.csv"),
            ("中药毒性摘要", "05_chinese_medicine_toxicity_summary.csv"),
            ("综合统计结果", "11_consolidated_count_summary.csv"),
        ],
        "tab_files": {
            "toxic_compounds": ("高危肝毒成分明细", "06_high_toxic_chemicals_master.csv"),
            "toxic_targets": ("毒性成分对应靶标", "07_high_toxic_chemical_targets.csv"),
            "toxic_pathways": ("毒性靶标-信号通路关系", "08_high_toxic_target_pathways.csv"),
            "toxic_go": ("毒性靶标-GO术语关系", "09_high_toxic_target_GO_terms.csv"),
            "toxic_diseases": ("毒性靶标-疾病关系", "10_high_toxic_target_disease_results.csv"),
        },
    },
    "compound": {
        "script": "03化学成分毒性预测_优化版_自动运行.py",
        "env_var": "TOXHERB_CHEM_CIDS_JSON",
        "summary_file": "05_chemical_toxicity_summary.csv",
        "livertox_file": "04_livertox_prediction_results.csv",
        "overview_files": [
            ("输出文件清单", "00_output_file_manifest.csv"),
            ("化学成分主表", "01_chemical_compound_master.csv"),
            ("化学成分来源中药与方剂", "02_chemical_source_herb_formulae.csv"),
            ("入血判定结果", "03_intoblood_prediction_results.csv"),
            ("肝毒性预测结果", "04_livertox_prediction_results.csv"),
            ("化学成分毒性摘要", "05_chemical_toxicity_summary.csv"),
            ("综合统计结果", "12_consolidated_count_summary.csv"),
        ],
        "tab_files": {
            "toxic_compounds": ("高危肝毒成分明细", "06_high_toxic_chemicals_master.csv"),
            "toxic_targets": ("毒性成分对应靶标", "07_high_toxic_chemical_targets.csv"),
            "toxic_pathways": ("毒性靶标-信号通路关系", "08_high_toxic_target_pathways.csv"),
            "toxic_go": ("毒性靶标-GO术语关系", "09_high_toxic_target_GO_terms.csv"),
            "toxic_diseases": ("毒性靶标-疾病关系", "10_high_toxic_target_disease_results.csv"),
        },
    },
}


def _top_counts(series: pd.Series, limit: int = 8) -> list[dict[str, Any]]:
    if series.empty:
        return []
    counts = series.dropna().astype(str).value_counts().head(limit)
    return [{"name": k, "count": int(v)} for k, v in counts.items()]


def _contains_any(series: pd.Series, items: list[str]) -> pd.Series:
    mask = pd.Series(False, index=series.index)
    clean = series.fillna("").astype(str)
    for item in items:
        mask |= clean.str.contains(re.escape(item), case=False, na=False)
    return mask


class HepatotoxicityPredictor:
    def __init__(self, root: str | Path | None = None) -> None:
        project_root = get_resource_root(Path(root) if root else None)
        self.paths = SystemPaths.from_root(project_root)
        self._data_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._ctd_cache_lock = threading.Lock()
        self._data: dict[str, pd.DataFrame] | None = None
        self._models: tuple[Any, Any, Any] | None = None
        self._warnings: list[str] = []

    def health(self) -> dict[str, Any]:
        missing_files = []
        for name, path in self.paths.__dict__.items():
            if name in {"root", "ctd_cache"}:
                continue
            if not Path(path).exists():
                missing_files.append(str(path))
        data_manifest = self._load_manifest(self.paths.root / "data_manifest.json")
        calibration_manifest = self._load_manifest(self.paths.root / "models" / "calibration" / "calibration_manifest.json")
        return {
            "ok": not missing_files,
            "root": str(self.paths.root),
            "missing_files": missing_files,
            "torch_available": torch is not None,
            "data_manifest": data_manifest,
            "model_manifest": self._model_manifest(),
            "calibration_manifest": calibration_manifest,
            "ctd_cache": {
                "path": str(self.paths.ctd_cache),
                "exists": self.paths.ctd_cache.exists(),
                "current": self._ctd_cache_is_current() if self.paths.ctd_cache.exists() else False,
            },
            "visualization_ready": True,
            "output_dir_status": {
                "user_dir": str(get_user_dir()),
                "prediction_outputs": str(Path(os.getenv("TOXHERB_OUTPUT_DIR", str(get_user_dir() / "prediction_outputs")))),
            },
        }

    def database_stats(self) -> dict[str, Any]:
        """
        实时统计数据库信息。
        注意：这里使用 refresh=True，确保每次请求都重新读取 CSV，
        避免前端看到的是程序启动时缓存的数据。
        """
        data = self._load_data(refresh=True)

        formula_herb = data["formula_herb"]
        herb_compound = data["herb_compound"]
        compound_class = data["compound_class"]
        compound_target = data["compound_target"]
        target_pathway = data["target_pathway"]
        go_term = data["go_term"]

        def count_unique(df: pd.DataFrame, column: str) -> int:
            if column not in df.columns:
                return 0
            s = df[column].dropna().astype(str).str.strip()
            s = s[s != ""]
            return int(s.nunique())

        def count_unique_first_existing(df: pd.DataFrame, columns: list[str]) -> int:
            for col in columns:
                if col in df.columns:
                    return count_unique(df, col)
            return int(len(df))

        formula_count = count_unique(formula_herb, "Formula.Chinese.name")

        herb_names = set()
        if "Herb.Chinese.name" in formula_herb.columns:
            herb_names.update(
                formula_herb["Herb.Chinese.name"]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist()
            )
        if "Herb.Chinese.name" in herb_compound.columns:
            herb_names.update(
                herb_compound["Herb.Chinese.name"]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist()
            )
        herb_names.discard("")
        herb_count = len(herb_names)

        return {
            "ok": True,
            "stats": {
                "formulas": formula_count,
                "herbs": herb_count,
                "compounds": count_unique_first_existing(compound_class, ["CID", "ChemicalName"]),
                "classes": count_unique(compound_class, "Class"),
                "superclasses": count_unique(compound_class, "Superclass"),
                "chemical_pathways": count_unique(compound_class, "Pathway"),
                "targets": count_unique(compound_target, "Symbol"),
                "signal_pathways": count_unique(target_pathway, "PathwayName"),
                "go_terms": count_unique_first_existing(go_term, ["GOID", "TERM"]),
                "diseases": self._ctd_cached_disease_count(),
            }
        }

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"status": "missing", "path": str(path)}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {"status": "invalid", "path": str(path), "error": str(exc)}

    def _model_manifest(self) -> dict[str, Any]:
        models = []
        for label, path in (
            ("cell", self.paths.model_cell),
            ("animal", self.paths.model_animal),
            ("clinical", self.paths.model_clinical),
        ):
            if path.exists():
                stat = path.stat()
                models.append(
                    {
                        "name": label,
                        "file_name": path.name,
                        "size_bytes": stat.st_size,
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
                        "hash": self._quick_file_hash(path),
                    }
                )
            else:
                models.append({"name": label, "file_name": path.name, "status": "missing"})
        return {"version": self._data_model_signature(), "models": models}

    @staticmethod
    def _quick_file_hash(path: Path, block_size: int = 1024 * 1024) -> str:
        if not path.exists():
            return "missing"
        stat = path.stat()
        hasher = hashlib.sha256()
        hasher.update(path.name.encode("utf-8", errors="ignore"))
        hasher.update(str(stat.st_size).encode("ascii"))
        hasher.update(str(int(stat.st_mtime_ns)).encode("ascii"))
        if stat.st_size:
            with path.open("rb") as handle:
                hasher.update(handle.read(block_size))
                if stat.st_size > block_size:
                    handle.seek(max(0, stat.st_size - block_size))
                    hasher.update(handle.read(block_size))
        return hasher.hexdigest()

    def _data_model_signature(self) -> str:
        paths = [
            self.paths.model_cell,
            self.paths.model_animal,
            self.paths.model_clinical,
            self.paths.formula_herb,
            self.paths.herb_compound,
            self.paths.compound_class,
            self.paths.compound_target,
            self.paths.target_pathway,
            self.paths.target_go,
            self.paths.go_term,
            self.paths.intoblood_reference,
            self.paths.food_medicine_homology,
            self.paths.ctd_cache,
        ]
        pieces = [self._quick_file_hash(path) for path in paths if path.exists()]
        return hashlib.sha256("|".join(pieces).encode("utf-8")).hexdigest()[:16]

    def input_hash(self, query_type: str, items: Iterable[Any]) -> str:
        payload = {
            "query_type": query_type,
            "items": [str(item).strip() for item in items if str(item).strip()],
        }
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def prediction_cache_key(self, query_type: str, items: Iterable[Any]) -> str:
        payload = {
            "input_hash": self.input_hash(query_type, items),
            "signature": self._data_model_signature(),
            "response_schema_version": PREDICTION_RESPONSE_SCHEMA_VERSION,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def predict_formula(self, text: str | Iterable[str]) -> dict[str, Any]:
        result, _ = self.run_prediction_pipeline("formula", text)
        return result

    def predict_herb(self, text: str | Iterable[str]) -> dict[str, Any]:
        result, _ = self.run_prediction_pipeline("herb", text)
        return result

    def predict_compound(self, text: str | Iterable[str]) -> dict[str, Any]:
        result, _ = self.run_prediction_pipeline("compound", text)
        return result

    def run_prediction_pipeline(
        self,
        query_type: str,
        text: str | Iterable[str],
        *,
        job_id: str | None = None,
        progress_callback: Callable[[str, int, str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        items = split_items(text)
        if not items:
            raise PredictionError("请输入至少一个检索目标。")
        if progress_callback:
            progress_callback("entity_resolution", 5, "正在解析输入实体。")

        if query_type == "formula":
            trace_chemicals, trace_context = self._resolve_formula(items)
            trace_details = self._build_external_trace_details("formula", trace_chemicals, trace_context)
            values: list[Any] = trace_context.get("matched_formulas") or items
        elif query_type == "herb":
            trace_chemicals, trace_context = self._resolve_herb(items)
            trace_details = self._build_external_trace_details("herb", trace_chemicals, trace_context)
            values = items
        elif query_type == "compound":
            trace_chemicals, trace_context = self._resolve_compound(items)
            trace_details = self._build_external_trace_details("compound", trace_chemicals, trace_context)
            values = self._resolve_external_compound_cids(items)
        else:
            raise PredictionError(f"不支持的预测类型: {query_type}", status_code=404)

        if cancel_event and cancel_event.is_set():
            raise PredictionError("任务已取消。", status_code=499)
        return self._run_external_prediction(
            query_type,
            values,
            items,
            trace_details,
            job_id=job_id,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    def _build_external_trace_details(
        self,
        query_type: str,
        chemicals: pd.DataFrame,
        context: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        if chemicals.empty:
            raise PredictionError("未在本地数据库中找到对应化学成分。")
        trace_frame = self._normalize_chemical_frame(chemicals)
        return self._build_trace_details(query_type, context, trace_frame)

    def _prediction_function_dir(self) -> Path:
        local_scripts = self.paths.root / "prediction_scripts"
        if local_scripts.exists():
            return local_scripts
        raise PredictionError(f"预测软件内置脚本目录不存在: {local_scripts}", status_code=500)

    def _resolve_external_compound_cids(self, items: list[str]) -> list[int]:
        resolved: list[int] = []
        unresolved: list[str] = []
        data: dict[str, pd.DataFrame] | None = None

        def add_cids(values: Iterable[Any]) -> None:
            for value in values:
                if pd.isna(value):
                    continue
                try:
                    cid = int(float(str(value).strip()))
                except (TypeError, ValueError):
                    continue
                if cid not in resolved:
                    resolved.append(cid)

        for item in items:
            text = item.strip()
            cid_match = re.fullmatch(r"(?i)(?:cid[\s:_-]*)?(\d+)", text)
            if cid_match:
                add_cids([cid_match.group(1)])
                continue

            if data is None:
                data = self._load_data()
            compounds = data["compound_class"]
            text_key = text.casefold()
            match = pd.DataFrame()
            for column in ("ChemicalName", "Smiles"):
                if column not in compounds.columns:
                    continue
                series = compounds[column].fillna("").astype(str)
                exact = compounds[series.str.strip().str.casefold() == text_key]
                if not exact.empty:
                    match = exact
                    break
            if match.empty and "ChemicalName" in compounds.columns:
                names = compounds["ChemicalName"].fillna("").astype(str)
                match = compounds[names.str.contains(re.escape(text), case=False, na=False)]

            if match.empty:
                unresolved.append(item)
            else:
                add_cids(match["CID"].dropna().unique().tolist())

        if unresolved:
            raise PredictionError(f"未能解析为 CID 的化学成分输入: {'、'.join(unresolved)}")
        if not resolved:
            raise PredictionError("未识别到有效的 CID、SMILES 或化学成分名称。")
        return resolved

    def _run_external_prediction(
        self,
        query_type: str,
        values: list[Any],
        original_query: list[str],
        trace_details: dict[str, list[dict[str, Any]]],
        *,
        job_id: str | None = None,
        progress_callback: Callable[[str, int, str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        config = EXTERNAL_PREDICTION_CONFIG[query_type]
        function_dir = self._prediction_function_dir()
        script_path = function_dir / config["script"]
        if not script_path.exists():
            raise PredictionError(f"预测脚本不存在: {script_path}", status_code=500)

        output_base = self._prediction_output_base(function_dir)
        run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        if job_id:
            run_id = f"{run_id}_{job_id[:8]}"
        env = os.environ.copy()
        env[config["env_var"]] = json.dumps(values, ensure_ascii=False)
        env["TOXHERB_OUTPUT_BASE_DIR"] = str(output_base)
        env["TOXHERB_RESOURCE_ROOT"] = str(self.paths.root)
        env["TOXHERB_RUN_ID"] = run_id
        env.setdefault("PYTHONIOENCODING", "utf-8")

        timeout = self._script_timeout_seconds()
        log_dir = output_base / "_job_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_log_path = log_dir / f"{job_id or run_id}.stdout.log"
        stderr_log_path = log_dir / f"{job_id or run_id}.stderr.log"
        try:
            completed = self.run_external_script(
                script_path,
                function_dir,
                env,
                timeout,
                stdout_log_path=stdout_log_path,
                stderr_log_path=stderr_log_path,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
        except subprocess.TimeoutExpired as exc:
            detail = self._tail_text((exc.stdout or "") + "\n" + (exc.stderr or ""))
            err = PredictionError(f"预测脚本运行超时（{timeout} 秒）: {config['script']}\n{detail}", status_code=504)
            err.stdout_log_path = str(stdout_log_path)
            err.stderr_log_path = str(stderr_log_path)
            raise err

        if completed.returncode != 0:
            if cancel_event and cancel_event.is_set():
                err = PredictionError("任务已取消。", status_code=499)
                err.stdout_log_path = str(stdout_log_path)
                err.stderr_log_path = str(stderr_log_path)
                raise err
            detail = self._tail_text(completed.stdout + "\n" + completed.stderr)
            err = PredictionError(
                f"预测脚本运行失败: {config['script']}，退出码 {completed.returncode}\n{detail}",
                status_code=500,
            )
            err.stdout_log_path = str(stdout_log_path)
            err.stderr_log_path = str(stderr_log_path)
            raise err

        output_dir = self._external_output_dir(completed.stdout + "\n" + completed.stderr, output_base, run_id)
        if progress_callback:
            progress_callback("response_building", 92, "正在读取预测输出并生成可视化数据。")
        try:
            response = self._build_external_prediction_response(
                query_type,
                original_query,
                output_dir,
                config,
                trace_details,
                job_id=job_id,
            )
        except Exception as exc:
            exc.output_dir = str(output_dir)
            exc.stdout_log_path = str(stdout_log_path)
            exc.stderr_log_path = str(stderr_log_path)
            raise
        return response, {
            "output_dir": str(output_dir),
            "stdout_log_path": str(stdout_log_path),
            "stderr_log_path": str(stderr_log_path),
        }

    def _prediction_output_base(self, function_dir: Path) -> Path:
        override = os.getenv("TOXHERB_OUTPUT_BASE_DIR")
        if override:
            output_base = Path(override)
        elif os.getenv("TOXHERB_OUTPUT_DIR"):
            output_base = Path(os.environ["TOXHERB_OUTPUT_DIR"])
        elif is_frozen_app():
            output_base = get_user_dir() / "prediction_outputs"
        else:
            output_base = get_user_dir() / "prediction_outputs"
        output_base.mkdir(parents=True, exist_ok=True)
        return output_base

    def run_external_script(
        self,
        script_path: Path,
        function_dir: Path,
        env: dict[str, str],
        timeout: int,
        *,
        stdout_log_path: Path,
        stderr_log_path: Path,
        progress_callback: Callable[[str, int, str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> subprocess.CompletedProcess[str]:
        python_cmd = os.getenv("TOXHERB_SCRIPT_PYTHON")
        if not python_cmd and is_frozen_app():
            completed = self._run_prediction_script_in_process(script_path, function_dir, env)
            stdout_log_path.write_text(completed.stdout, encoding="utf-8")
            stderr_log_path.write_text(completed.stderr, encoding="utf-8")
            self._replay_progress_markers(completed.stdout + "\n" + completed.stderr, progress_callback)
            return completed

        command = [python_cmd or sys.executable, str(script_path)]
        started = time.time()
        process = subprocess.Popen(
            command,
            cwd=function_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        line_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def pump(stream: Any, stream_name: str, log_path: Path) -> None:
            with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
                if stream is None:
                    return
                for line in iter(stream.readline, ""):
                    log_file.write(line)
                    log_file.flush()
                    line_queue.put((stream_name, line))
                stream.close()

        threads = [
            threading.Thread(target=pump, args=(process.stdout, "stdout", stdout_log_path), daemon=True),
            threading.Thread(target=pump, args=(process.stderr, "stderr", stderr_log_path), daemon=True),
        ]
        for thread in threads:
            thread.start()

        cancelled = False
        while process.poll() is None or not line_queue.empty():
            if cancel_event and cancel_event.is_set() and process.poll() is None:
                cancelled = True
                process.terminate()
            if time.time() - started > timeout and process.poll() is None:
                process.kill()
                raise subprocess.TimeoutExpired(command, timeout, "\n".join(stdout_lines), "\n".join(stderr_lines))
            try:
                stream_name, line = line_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if stream_name == "stdout":
                stdout_lines.append(line)
            else:
                stderr_lines.append(line)
            self._parse_progress_line(line, progress_callback)

        for thread in threads:
            thread.join(timeout=1)
        while not line_queue.empty():
            stream_name, line = line_queue.get_nowait()
            if stream_name == "stdout":
                stdout_lines.append(line)
            else:
                stderr_lines.append(line)
            self._parse_progress_line(line, progress_callback)

        returncode = process.returncode
        if cancelled and returncode == 0:
            returncode = 1
        return subprocess.CompletedProcess(command, returncode, "".join(stdout_lines), "".join(stderr_lines))

    def _execute_prediction_script(
        self,
        script_path: Path,
        function_dir: Path,
        env: dict[str, str],
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        python_cmd = os.getenv("TOXHERB_SCRIPT_PYTHON")
        if python_cmd or not is_frozen_app():
            return subprocess.run(
                [python_cmd or sys.executable, str(script_path)],
                cwd=function_dir,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        return self._run_prediction_script_in_process(script_path, function_dir, env)

    def _run_prediction_script_in_process(
        self,
        script_path: Path,
        function_dir: Path,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        old_env = os.environ.copy()
        old_cwd = Path.cwd()
        old_argv = sys.argv[:]
        old_path = sys.path[:]
        stdout = io.StringIO()
        stderr = io.StringIO()
        returncode = 0
        try:
            os.environ.clear()
            os.environ.update(env)
            os.chdir(function_dir)
            sys.argv = [str(script_path)]
            if str(function_dir) not in sys.path:
                sys.path.insert(0, str(function_dir))
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                runpy.run_path(str(script_path), run_name="__main__")
        except SystemExit as exc:
            code = exc.code
            if code is None:
                returncode = 0
            elif isinstance(code, int):
                returncode = code
            else:
                returncode = 1
                print(code, file=stderr)
        except Exception:
            returncode = 1
            traceback.print_exc(file=stderr)
        finally:
            os.environ.clear()
            os.environ.update(old_env)
            os.chdir(old_cwd)
            sys.argv = old_argv
            sys.path = old_path
        return subprocess.CompletedProcess(
            [str(script_path)],
            returncode,
            stdout.getvalue(),
            stderr.getvalue(),
        )

    @staticmethod
    def _script_timeout_seconds() -> int:
        raw = os.getenv("TOXHERB_SCRIPT_TIMEOUT_SECONDS", "3600")
        try:
            return max(30, int(raw))
        except ValueError:
            return 3600

    @staticmethod
    def _parse_progress_line(
        line: str,
        progress_callback: Callable[[str, int, str], None] | None = None,
    ) -> None:
        if not progress_callback:
            return
        marker = "TOXHERB_PROGRESS="
        if marker not in line:
            return
        payload = line.split(marker, 1)[1].strip()
        parts = payload.split("|", 2)
        if len(parts) != 3:
            return
        stage, percent_text, message = parts
        try:
            percent = int(float(percent_text))
        except ValueError:
            percent = 0
        progress_callback(stage.strip() or "running", percent, message.strip())

    def _replay_progress_markers(
        self,
        text: str,
        progress_callback: Callable[[str, int, str], None] | None = None,
    ) -> None:
        for line in text.splitlines():
            self._parse_progress_line(line, progress_callback)

    @staticmethod
    def _tail_text(text: str, max_lines: int = 80) -> str:
        lines = [line for line in text.splitlines() if line.strip()]
        return "\n".join(lines[-max_lines:])

    def _external_output_dir(self, output_text: str, output_base: Path, run_id: str) -> Path:
        for line in output_text.splitlines():
            if line.startswith("TOXHERB_OUTPUT_DIR="):
                path = Path(line.split("=", 1)[1].strip())
                if path.exists():
                    return path
        candidates = sorted(
            output_base.glob(f"*_{run_id}"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse=True,
        )
        if candidates:
            return candidates[0]
        raise PredictionError(f"未能定位本次预测输出目录，RUN_ID={run_id}", status_code=500)

    @staticmethod
    def _json_safe_value(value: Any) -> Any:
        native = _native(value)
        if isinstance(native, dict):
            return {str(key): HepatotoxicityPredictor._json_safe_value(item) for key, item in native.items()}
        if isinstance(native, (list, tuple, set)):
            return [HepatotoxicityPredictor._json_safe_value(item) for item in native]
        if native is None or isinstance(native, (str, int, float, bool)):
            return native
        return str(native)

    def _read_external_csv(self, path: Path, **kwargs: Any) -> pd.DataFrame:
        if not path.exists():
            raise PredictionError(f"预测输出缺少 CSV 文件: {path}", status_code=500)
        kwargs.setdefault("low_memory", False)
        return _read_csv(path, **kwargs)

    def _section_from_csv(self, title: str, path: Path, section_id: str | None = None) -> dict[str, Any]:
        preview_limit = EXTERNAL_CSV_SECTION_ROW_LIMIT
        df = self._read_external_csv(path, nrows=preview_limit + 1)
        truncated = len(df) > preview_limit
        if truncated:
            df = df.head(preview_limit)
        df = _normalize_relation_output_frame(df, section_id)
        if section_id == "toxic_compounds":
            df = self._fill_compound_names_from_cid(df)
        if section_id in {"toxic_targets", "toxic_pathways", "toxic_diseases"}:
            df = self._fill_target_identifiers(df)
            df = _normalize_relation_output_frame(df, section_id)
        actual_row_count = self._manifest_row_count(path.parent, path.name)
        rows = [
            [self._json_safe_value(value) for value in row]
            for row in df.itertuples(index=False, name=None)
        ]
        return {
            "title": title,
            "headers": [str(column) for column in df.columns],
            "rows": rows,
            "source_file": path.name,
            "row_count": actual_row_count if actual_row_count is not None else len(rows),
            "truncated": truncated or (actual_row_count is not None and actual_row_count > preview_limit),
            "preview_limit": preview_limit,
        }

    def _section_from_dataframe(
        self,
        title: str,
        df: pd.DataFrame,
        *,
        source_file: str,
        section_id: str | None = None,
    ) -> dict[str, Any]:
        preview_limit = EXTERNAL_CSV_SECTION_ROW_LIMIT
        display = _normalize_relation_output_frame(df, section_id)
        preview = display.head(preview_limit)
        rows = [
            [self._json_safe_value(value) for value in row]
            for row in preview.itertuples(index=False, name=None)
        ]
        return {
            "title": title,
            "headers": [str(column) for column in preview.columns],
            "rows": rows,
            "source_file": source_file,
            "row_count": int(len(display)),
            "truncated": len(display) > preview_limit,
            "preview_limit": preview_limit,
        }

    def _records_from_external_df(self, df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        source = df.head(limit) if limit is not None else df
        for row in source.to_dict(orient="records"):
            records.append({str(key): self._json_safe_value(value) for key, value in row.items()})
        return records

    def _fill_compound_names_from_cid(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        out = df.copy()
        if "ChemicalName" not in out.columns:
            out["ChemicalName"] = pd.NA
        if "CID" not in out.columns and "Chemical_component_group" in out.columns:
            extracted = out["Chemical_component_group"].fillna("").astype(str).str.extract(r"CID[_:\-\s]*(\d+)", expand=False)
            out["CID"] = extracted
        if "CID" not in out.columns:
            return out
        missing = _blank_mask(out["ChemicalName"])
        if not missing.any():
            return out
        try:
            data = self._load_data()
        except (AttributeError, KeyError, PredictionError):
            return out
        compounds = self._select_columns(data.get("compound_class", pd.DataFrame()), ["CID", "ChemicalName"])
        if compounds.empty:
            return out
        compounds["CID_key"] = pd.to_numeric(compounds["CID"], errors="coerce")
        compounds = compounds.dropna(subset=["CID_key"]).drop_duplicates(subset=["CID_key"])
        fill_base = out.loc[missing].copy()
        fill_base["CID_key"] = pd.to_numeric(fill_base["CID"], errors="coerce")
        fill_base = fill_base.merge(
            compounds[["CID_key", "ChemicalName"]].rename(columns={"ChemicalName": "ChemicalName_fill"}),
            on="CID_key",
            how="left",
        )
        out.loc[missing, "ChemicalName"] = fill_base["ChemicalName_fill"].to_numpy()
        return out

    def _disease_rows_for_symbols_from_data_file(
        self,
        symbols: Iterable[str],
        entrez_ids: Iterable[Any] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        keys = sorted({self._symbol_key(symbol) for symbol in symbols if self._symbol_key(symbol)})
        entrez_keys = sorted({str(_identifier_text(value) or "").strip() for value in (entrez_ids or []) if str(_identifier_text(value) or "").strip()})
        columns = ["Symbol", "ENTREZID", "DiseaseName", "DiseaseID"]
        if not keys and not entrez_keys:
            return pd.DataFrame(columns=columns)
        frames: list[pd.DataFrame] = []
        selected_count = 0
        for chunk in _read_csv_chunks_with_aliases(
            self.paths.target_disease,
            columns,
            EXTERNAL_SECTION_CHUNK_SIZE,
            dtype=str,
        ):
            chunk["symbol_key"] = chunk["Symbol"].map(self._symbol_key)
            chunk["entrez_key"] = chunk["ENTREZID"].map(_identifier_text).fillna("").astype(str)
            mask = pd.Series(False, index=chunk.index)
            if keys:
                mask |= chunk["symbol_key"].isin(keys)
            if entrez_keys:
                mask |= chunk["entrez_key"].isin(entrez_keys)
            matched = chunk[mask].copy()
            if matched.empty:
                continue
            selected = self._select_columns(matched, columns)
            selected = _drop_blank_relation_rows(selected, ["DiseaseName", "DiseaseID"])
            if selected.empty:
                continue
            frames.append(selected)
            selected_count += len(selected)
            if limit is not None and selected_count >= limit:
                break
        if not frames:
            return pd.DataFrame(columns=columns)
        out = pd.concat(frames, ignore_index=True).drop_duplicates()
        return out.head(limit).copy() if limit is not None else out

    def _fill_target_link_chemical_names(self, target_links: pd.DataFrame) -> pd.DataFrame:
        if target_links.empty:
            return target_links.copy()
        out = _coalesce_relation_columns(target_links)
        for column in ("ChemicalName", "Symbol", "ENTREZID"):
            if column not in out.columns:
                out[column] = pd.NA
        out["ENTREZID"] = out["ENTREZID"].map(_identifier_text)

        data = self._load_data()
        missing_symbol = _blank_mask(out["Symbol"]) & ~_blank_mask(out["ENTREZID"])
        if missing_symbol.any():
            target_pathway = self._select_columns(data.get("target_pathway", pd.DataFrame()), ["ENTREZID", "Symbol"])
            if not target_pathway.empty:
                target_pathway["ENTREZID"] = target_pathway["ENTREZID"].map(_identifier_text)
                target_pathway = _drop_blank_relation_rows(target_pathway, ["ENTREZID", "Symbol"]).drop_duplicates()
                symbol_filled = (
                    out.loc[missing_symbol].drop(columns=["Symbol"])
                    .merge(target_pathway, on="ENTREZID", how="left")
                )
                out = pd.concat([out.loc[~missing_symbol], symbol_filled], ignore_index=True, sort=False)

        out = self._select_columns(out, ["ChemicalName", "Symbol", "ENTREZID"])
        missing_mask = _blank_mask(out["ChemicalName"])
        if not missing_mask.any():
            return out[~_blank_mask(out["Symbol"])].drop_duplicates().reset_index(drop=True)

        symbols = out.loc[missing_mask, "Symbol"].dropna().astype(str).unique().tolist()
        if not symbols:
            return out[~_blank_mask(out["ChemicalName"]) & ~_blank_mask(out["Symbol"])].drop_duplicates().reset_index(drop=True)
        compound_target = self._select_columns(data.get("compound_target", pd.DataFrame()), ["ChemicalName", "Symbol"])
        if compound_target.empty:
            return out[~_blank_mask(out["ChemicalName"]) & ~_blank_mask(out["Symbol"])].drop_duplicates().reset_index(drop=True)
        symbol_keys = {self._symbol_key(symbol) for symbol in symbols}
        compound_target["symbol_key"] = compound_target["Symbol"].map(self._symbol_key)
        compound_target = compound_target[compound_target["symbol_key"].isin(symbol_keys)].copy()
        compound_target = _drop_blank_relation_rows(compound_target, ["ChemicalName", "Symbol"])
        if compound_target.empty:
            return out[~_blank_mask(out["ChemicalName"]) & ~_blank_mask(out["Symbol"])].drop_duplicates().reset_index(drop=True)

        existing = out[~missing_mask].copy()
        missing_base = out[missing_mask].drop(columns=["ChemicalName"])
        missing_base["symbol_key"] = missing_base["Symbol"].map(self._symbol_key)
        filled = missing_base.merge(
            compound_target[["symbol_key", "ChemicalName"]],
            on="symbol_key",
            how="inner",
        ).drop(columns=["symbol_key"])
        combined = pd.concat([existing, filled], ignore_index=True, sort=False)
        combined = combined[~_blank_mask(combined["ChemicalName"]) & ~_blank_mask(combined["Symbol"])]
        return combined.drop_duplicates().reset_index(drop=True)

    def _fill_target_identifiers(self, rows: pd.DataFrame) -> pd.DataFrame:
        if rows.empty:
            return rows.copy()
        out = _coalesce_relation_columns(rows)
        if "Symbol" not in out.columns and "ENTREZID" not in out.columns:
            return out
        for column in ("Symbol", "ENTREZID"):
            if column not in out.columns:
                out[column] = pd.NA
        out["ENTREZID"] = out["ENTREZID"].map(_identifier_text)

        try:
            data = self._load_data()
        except (AttributeError, KeyError, PredictionError):
            return out
        pathway = self._select_columns(data.get("target_pathway", pd.DataFrame()), ["Symbol", "ENTREZID"])
        if pathway.empty:
            return out
        pathway["ENTREZID"] = pathway["ENTREZID"].map(_identifier_text)
        pathway = _drop_blank_relation_rows(pathway, ["Symbol", "ENTREZID"]).drop_duplicates()
        pathway["symbol_key"] = pathway["Symbol"].map(self._symbol_key)
        pathway["entrez_key"] = pathway["ENTREZID"].fillna("").astype(str)

        missing_entrez = _blank_mask(out["ENTREZID"]) & ~_blank_mask(out["Symbol"])
        if missing_entrez.any():
            by_symbol = pathway.drop_duplicates(subset=["symbol_key"])[["symbol_key", "ENTREZID"]].rename(columns={"ENTREZID": "ENTREZID_fill"})
            fill_base = out.loc[missing_entrez].copy()
            fill_base["symbol_key"] = fill_base["Symbol"].map(self._symbol_key)
            fill_base = fill_base.merge(by_symbol, on="symbol_key", how="left").drop(columns=["symbol_key"])
            out.loc[missing_entrez, "ENTREZID"] = fill_base["ENTREZID_fill"].to_numpy()

        missing_symbol = _blank_mask(out["Symbol"]) & ~_blank_mask(out["ENTREZID"])
        if missing_symbol.any():
            by_entrez = pathway.drop_duplicates(subset=["entrez_key"])[["entrez_key", "Symbol"]].rename(columns={"Symbol": "Symbol_fill"})
            fill_base = out.loc[missing_symbol].copy()
            fill_base["entrez_key"] = fill_base["ENTREZID"].fillna("").astype(str)
            fill_base = fill_base.merge(by_entrez, on="entrez_key", how="left").drop(columns=["entrez_key"])
            out.loc[missing_symbol, "Symbol"] = fill_base["Symbol_fill"].to_numpy()

        return out

    def _target_disease_frame_from_targets(self, target_df: pd.DataFrame, fallback_df: pd.DataFrame | None = None) -> pd.DataFrame:
        fallback_source = fallback_df if fallback_df is not None else pd.DataFrame()
        fallback = _filter_toxic_disease_rows(fallback_source)
        target_links = self._select_columns(target_df, ["ChemicalName", "Symbol", "ENTREZID"])
        target_links = self._fill_target_identifiers(target_links)
        target_links = self._fill_target_link_chemical_names(target_links)
        if target_links.empty or "Symbol" not in target_links.columns:
            return fallback
        symbols = target_links["Symbol"].dropna().astype(str).unique().tolist()
        entrez_ids = target_links["ENTREZID"].dropna().astype(str).unique().tolist() if "ENTREZID" in target_links.columns else []
        try:
            disease_rows = self._disease_rows_for_symbols(symbols)
            disease_rows = _drop_blank_relation_rows(disease_rows, ["DiseaseName", "DiseaseID"])
        except (PredictionError, OSError, sqlite3.Error):
            disease_rows = pd.DataFrame(columns=["Symbol", "ENTREZID", "DiseaseName", "DiseaseID"])
        if disease_rows.empty:
            disease_rows = self._disease_rows_for_symbols_from_data_file(symbols, entrez_ids)
        if disease_rows.empty:
            return fallback
        left_by_symbol = target_links.copy()
        right_by_symbol = disease_rows.copy()
        left_by_symbol["symbol_key"] = left_by_symbol["Symbol"].map(self._symbol_key)
        right_by_symbol["symbol_key"] = right_by_symbol["Symbol"].map(self._symbol_key)
        links = left_by_symbol.merge(right_by_symbol, on="symbol_key", how="inner", suffixes=("", "_disease"))
        links = _coalesce_relation_columns(links)
        if links.empty and "ENTREZID" in target_links.columns and "ENTREZID" in disease_rows.columns:
            left = target_links.copy()
            right = disease_rows.copy()
            left["ENTREZID"] = left["ENTREZID"].map(_identifier_text)
            right["ENTREZID"] = right["ENTREZID"].map(_identifier_text)
            links = left.merge(right, on="ENTREZID", how="inner", suffixes=("", "_disease"))
        links = self._select_columns(links, SECTION_CORE_COLUMNS["toxic_diseases"])
        links = _filter_toxic_disease_rows(links)
        if links.empty:
            return fallback
        return links.drop_duplicates().reset_index(drop=True)

    def _build_external_prediction_response(
        self,
        query_type: str,
        query: list[str],
        output_dir: Path,
        config: dict[str, Any],
        trace_details: dict[str, list[dict[str, Any]]] | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        summary_df = self._read_external_csv(output_dir / config["summary_file"])
        livertox_df = self._read_external_csv(output_dir / config["livertox_file"])
        high_toxic_df = self._read_external_csv(output_dir / config["tab_files"]["toxic_compounds"][1])
        target_df = self._read_external_csv(output_dir / config["tab_files"]["toxic_targets"][1])
        pathway_df = self._read_external_csv(output_dir / config["tab_files"]["toxic_pathways"][1])
        go_df = self._read_external_csv(output_dir / config["tab_files"]["toxic_go"][1])
        disease_df = self._read_external_csv(
            output_dir / config["tab_files"]["toxic_diseases"][1],
            nrows=EXTERNAL_RESPONSE_RECORD_LIMIT,
        )
        livertox_df, high_toxic_df, target_df, pathway_df, go_df, disease_df = self._augment_external_prediction_frames(
            livertox_df,
            high_toxic_df,
            target_df,
            pathway_df,
            go_df,
            disease_df,
        )
        target_df = _normalize_relation_output_frame(target_df)
        pathway_df = _normalize_relation_output_frame(pathway_df)
        go_df = _normalize_relation_output_frame(go_df)
        disease_df = self._target_disease_frame_from_targets(target_df, disease_df)
        disease_df = self._add_disease_priority(_normalize_relation_output_frame(disease_df, "toxic_diseases"))
        pathway_display_df = _normalize_relation_output_frame(pathway_df, "toxic_pathways")
        go_display_df = _normalize_relation_output_frame(go_df, "toxic_go")
        disease_display_df = _normalize_relation_output_frame(disease_df, "toxic_diseases")

        csv_sections: dict[str, list[dict[str, Any]]] = {
            "overview": [
                self._section_from_csv(title, output_dir / filename)
                for title, filename in config["overview_files"]
            ]
        }
        csv_sections["toxic_compounds"] = [
            self._section_from_csv(config["tab_files"]["toxic_compounds"][0], output_dir / config["tab_files"]["toxic_compounds"][1], "toxic_compounds")
        ]
        csv_sections["toxic_targets"] = [
            self._section_from_dataframe(
                config["tab_files"]["toxic_targets"][0],
                target_df,
                source_file=config["tab_files"]["toxic_targets"][1],
                section_id="toxic_targets",
            )
        ]
        csv_sections["toxic_pathways"] = [
            self._section_from_dataframe(
                config["tab_files"]["toxic_pathways"][0],
                pathway_display_df,
                source_file=config["tab_files"]["toxic_pathways"][1],
                section_id="toxic_pathways",
            )
        ]
        csv_sections["toxic_go"] = [
            self._section_from_dataframe(
                config["tab_files"]["toxic_go"][0],
                go_display_df,
                source_file=config["tab_files"]["toxic_go"][1],
                section_id="toxic_go",
            )
        ]
        csv_sections["toxic_diseases"] = [
            {
                "title": config["tab_files"]["toxic_diseases"][0],
                "headers": [str(column) for column in disease_display_df.head(EXTERNAL_CSV_SECTION_ROW_LIMIT).columns],
                "rows": [
                    [self._json_safe_value(value) for value in row]
                    for row in disease_display_df.head(EXTERNAL_CSV_SECTION_ROW_LIMIT).itertuples(index=False, name=None)
                ],
                "source_file": config["tab_files"]["toxic_diseases"][1],
                "row_count": int(len(disease_df)),
                "truncated": len(disease_df) > EXTERNAL_CSV_SECTION_ROW_LIMIT,
                "preview_limit": EXTERNAL_CSV_SECTION_ROW_LIMIT,
            }
        ]

        summary_row = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
        probabilities = {
            "cell": self._numeric_max(livertox_df, "Pred_Cell_prob"),
            "animal": self._numeric_max(livertox_df, "Pred_Animal_prob"),
            "clinical": self._numeric_max(livertox_df, "Pred_Clinical_prob"),
            "max": self._numeric_max(livertox_df, "Max_Tox_Prob"),
        }
        threshold = self._float_value(summary_row.get("Toxic_probability_threshold"), 0.85)
        high_count = self._int_value(summary_row.get("High_toxic_molecule_count"), len(high_toxic_df))
        enrichment = self._float_value(summary_row.get("Toxic_enrichment_ratio"), 0.0)
        absorbed_count = self._absorbed_count(summary_row.get("Absorbed_molecule_count_or_percent"))
        risk_label = str(summary_row.get("Risk_level") or "已完成")
        intoblood_count = (
            int((pd.to_numeric(livertox_df["IntoBlood"], errors="coerce") == 1).sum())
        if "IntoBlood" in livertox_df.columns
            else 0
        )
        herb_candidates = self._summary_herb_candidates(query_type, query, trace_details or {}, livertox_df)
        food_medicine_homology = self._food_medicine_homology_summary(herb_candidates)
        consensus_reference = self._consensus_reference(livertox_df, absorbed_count)
        confidence_profile = self._build_confidence_profile(livertox_df, summary_row, probabilities)
        domain_profile = self._build_applicability_domain(livertox_df)
        top_compound_names = self._top_names(high_toxic_df, "ChemicalName", "compound_priority_score", 8)
        top_target_names = self._top_names(target_df, "Symbol", "target_priority_score", 8)
        top_pathway_names = self._top_names(pathway_df, "PathwayName", "pathway_priority_score", 8)

        summary = {
            "total_compounds": int(len(livertox_df)),
            "valid_smiles_count": self._int_value(summary_row.get("Original_valid_molecule_count"), int(len(livertox_df))),
            "intoblood_count": intoblood_count,
            "absorbed_count": absorbed_count,
            "toxic_threshold": threshold,
            "high_toxic_count": high_count,
            "enrichment_ratio": enrichment,
            "max_toxic_probability": probabilities["max"],
            "raw_max_toxic_probability": probabilities["max"],
            "calibrated_max_toxic_probability": probabilities["max"],
            "calibration_status": "embedded",
            "calibration_version": "model_bundle_artifact_v1",
            "confidence_level": confidence_profile["confidence_level"],
            "confidence_score": confidence_profile["confidence_score"],
            "confidence_reasons": confidence_profile["confidence_reasons"],
            "domain_label": domain_profile["domain_label"],
            "domain_coverage": domain_profile["domain_coverage"],
            "exposure_weighted_risk": self._exposure_weighted_risk(livertox_df),
            "top_compound_names": top_compound_names,
            "top_target_names": top_target_names,
            "top_pathway_names": top_pathway_names,
            "disclaimer": DISCLAIMER_TEXT,
            "risk_level": self._risk_level_from_label(risk_label),
            "risk_label": risk_label,
            "advice": str(summary_row.get("Advice") or "无"),
            "output_dir": str(output_dir),
            "food_medicine_homology": food_medicine_homology,
            "consensus_reference": consensus_reference,
        }
        summary["confidence_profile"] = confidence_profile
        summary["applicability_domain"] = domain_profile

        high_records = self._records_from_external_df(high_toxic_df, EXTERNAL_RESPONSE_RECORD_LIMIT)
        target_records = self._records_from_external_df(target_df, EXTERNAL_RESPONSE_RECORD_LIMIT)
        pathway_records = self._records_from_external_df(pathway_display_df, EXTERNAL_RESPONSE_RECORD_LIMIT)
        go_records = self._records_from_external_df(go_display_df, EXTERNAL_RESPONSE_RECORD_LIMIT)
        disease_records = self._records_from_external_df(disease_display_df, EXTERNAL_RESPONSE_RECORD_LIMIT)
        sections = self._build_section_index(query_type, output_dir, config, job_id)
        for section in sections:
            if section.get("id") == "toxic_targets":
                section["row_count"] = int(len(target_df))
                section["truncated"] = len(target_df) > EXTERNAL_CSV_SECTION_ROW_LIMIT
            if section.get("id") == "toxic_pathways":
                section["row_count"] = int(len(pathway_display_df))
                section["truncated"] = len(pathway_display_df) > EXTERNAL_CSV_SECTION_ROW_LIMIT
            if section.get("id") == "toxic_go":
                section["row_count"] = int(len(go_display_df))
                section["truncated"] = len(go_display_df) > EXTERNAL_CSV_SECTION_ROW_LIMIT
            if section.get("id") == "toxic_diseases":
                section["row_count"] = int(len(disease_df))
                section["truncated"] = len(disease_df) > EXTERNAL_CSV_SECTION_ROW_LIMIT

        response = {
            "ok": True,
            "response_schema_version": PREDICTION_RESPONSE_SCHEMA_VERSION,
            "query_type": query_type,
            "query": query,
            "summary": summary,
            "probabilities": probabilities,
            "context": {"output_dir": str(output_dir)},
            "warnings": [],
            "class_counts": self._count_records(high_toxic_df, "Class"),
            "superclass_counts": self._count_records(high_toxic_df, "Superclass"),
            "pathway_family_counts": self._count_records(high_toxic_df, "Pathway"),
            "target_counts": self._count_records(target_df, "Symbol"),
            "pathway_counts": self._count_records(pathway_df, "PathwayName"),
            "go_counts": self._count_records(go_df, "TERM"),
            "high_risk_compounds": high_records,
            "compounds": self._records_from_external_df(livertox_df, EXTERNAL_COMPOUND_RECORD_LIMIT),
            "trace_details": trace_details or {},
            "toxicity_details": {
                "toxic_compounds": high_records,
                "toxic_targets": target_records,
                "toxic_target_go": go_records,
                "toxic_target_pathways": pathway_records,
                "toxic_target_diseases": disease_records,
            },
            "csv_sections": csv_sections,
            "sections": sections,
        }
        response["chart_specs"] = self.build_prediction_visuals_from_frames(
            summary,
            probabilities,
            high_toxic_df,
            target_df,
            pathway_df,
            go_df,
            disease_df,
        )
        return response

    @staticmethod
    def _numeric_max(df: pd.DataFrame, column: str) -> float:
        if column not in df.columns:
            return 0.0
        values = pd.to_numeric(df[column], errors="coerce")
        values = values[values >= 0]
        if values.empty:
            return 0.0
        return round(float(values.max()), 4)

    @staticmethod
    def _float_value(value: Any, default: float) -> float:
        try:
            if value is None or pd.isna(value):
                return default
            number = float(value)
            return number if math.isfinite(number) else default
        except (TypeError, ValueError):
            match = re.search(r"-?\d+(?:\.\d+)?", str(value))
            if not match:
                return default
            number = float(match.group(0))
            return number if math.isfinite(number) else default

    @staticmethod
    def _int_value(value: Any, default: int) -> int:
        try:
            if value is None or pd.isna(value):
                return default
            number = float(value)
            return int(number) if math.isfinite(number) else default
        except (TypeError, ValueError):
            match = re.search(r"\d+", str(value))
            return int(match.group(0)) if match else default

    def _absorbed_count(self, value: Any) -> int:
        return self._int_value(value, 0)

    @staticmethod
    def _risk_level_from_label(label: str) -> str:
        text = str(label).casefold()
        if "high" in text or "高" in text or "🔴" in text:
            return "high"
        if "moderate" in text or "medium" in text or "中" in text or "🟡" in text:
            return "moderate"
        return "low"

    @staticmethod
    def _count_records(df: pd.DataFrame, column: str, limit: int = 8) -> list[dict[str, Any]]:
        if column not in df.columns:
            return []
        return _top_counts(df[column], limit)

    def _augment_external_prediction_frames(
        self,
        livertox_df: pd.DataFrame,
        high_toxic_df: pd.DataFrame,
        target_df: pd.DataFrame,
        pathway_df: pd.DataFrame,
        go_df: pd.DataFrame,
        disease_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        livertox_df = self._fill_compound_names_from_cid(livertox_df)
        high_toxic_df = self._fill_compound_names_from_cid(high_toxic_df)
        livertox_df = self._add_compound_priority(livertox_df)
        high_toxic_df = self._add_compound_priority(high_toxic_df)
        target_df = _coalesce_relation_columns(target_df)
        pathway_df = _coalesce_relation_columns(pathway_df)
        go_df = _coalesce_relation_columns(go_df)
        disease_df = _coalesce_relation_columns(disease_df)
        if target_df.empty:
            target_df = self._target_rows_for_high_toxic(high_toxic_df)
        target_df = self._fill_target_identifiers(target_df)
        if pathway_df.empty and not target_df.empty:
            pathway_df = self._pathway_rows_for_targets(target_df)
        if go_df.empty and not target_df.empty:
            go_df = self._go_rows_for_targets(target_df)
        pathway_df = self._fill_target_identifiers(pathway_df)
        disease_df = self._fill_target_identifiers(disease_df)
        target_df = self._add_target_priority(target_df, high_toxic_df)
        pathway_df = self._add_pathway_priority(pathway_df)
        go_df = self._add_go_priority(go_df)
        disease_df = self._add_disease_priority(disease_df)
        return livertox_df, high_toxic_df, target_df, pathway_df, go_df, disease_df

    def _add_compound_priority(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        out = df.copy()
        risk = self._numeric_series(out, "Max_Tox_Prob")
        bioavailability = self._numeric_series(out, "Bioavailability_Ma")
        intoblood = self._numeric_series(out, "IntoBlood")
        absorbed = self._numeric_series(out, "Is_Absorbed")
        source_count = self._source_count_series(out)
        domain_score = self._domain_score_series(out)
        out["domain_label"] = self._domain_label_series(out)
        out["domain_score"] = domain_score.round(4)
        out["compound_priority_score"] = (
            risk.fillna(0) * 0.45
            + bioavailability.fillna(0) * 0.15
            + intoblood.fillna(0).clip(0, 1) * 0.12
            + absorbed.fillna(0).clip(0, 1) * 0.12
            + source_count.fillna(0).clip(0, 8) / 8 * 0.08
            + domain_score.fillna(0.5) * 0.08
        ).round(4)
        return out.sort_values("compound_priority_score", ascending=False)

    @staticmethod
    def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
        if column not in df.columns:
            return pd.Series(0.0, index=df.index, dtype=float)
        return pd.to_numeric(df[column], errors="coerce")

    @staticmethod
    def _source_count_series(df: pd.DataFrame) -> pd.Series:
        for column in ("Source_Herb_Count", "Source_Formulas_Count"):
            if column in df.columns:
                return pd.to_numeric(df[column], errors="coerce").fillna(0)
        if "Source_Herbs" in df.columns:
            return df["Source_Herbs"].fillna("").astype(str).map(
                lambda value: len([part for part in re.split(r"[;；,，、]+", value) if part.strip()])
            )
        if "Herb.Chinese.name" in df.columns:
            return df["Herb.Chinese.name"].fillna("").astype(str).map(lambda value: 1 if value.strip() else 0)
        return pd.Series(0.0, index=df.index, dtype=float)

    @staticmethod
    def _domain_label_series(df: pd.DataFrame) -> pd.Series:
        if "Smiles" not in df.columns:
            return pd.Series("insufficient", index=df.index)
        smiles = df["Smiles"].fillna("").astype(str).str.strip()
        labels = pd.Series("borderline", index=df.index)
        labels[smiles == ""] = "insufficient"
        if "Class" in df.columns:
            known_class = df["Class"].fillna("").astype(str).str.strip() != ""
            labels[(smiles != "") & known_class] = "in_domain"
        return labels

    def _domain_score_series(self, df: pd.DataFrame) -> pd.Series:
        labels = self._domain_label_series(df)
        return labels.map({"in_domain": 1.0, "borderline": 0.65, "out_of_domain": 0.25, "insufficient": 0.2}).astype(float)

    def _add_target_priority(self, target_df: pd.DataFrame, high_toxic_df: pd.DataFrame) -> pd.DataFrame:
        if target_df.empty:
            return target_df.copy()
        out = target_df.copy()
        if "ChemicalName" in out.columns and "ChemicalName" in high_toxic_df.columns:
            toxic_meta = high_toxic_df[[col for col in ["ChemicalName", "Max_Tox_Prob", "compound_priority_score"] if col in high_toxic_df.columns]]
            toxic_meta = toxic_meta.drop_duplicates(subset=["ChemicalName"])
            out = out.merge(toxic_meta, on="ChemicalName", how="left", suffixes=("", "_compound"))
        degree = out["Symbol"].map(out["Symbol"].value_counts()) if "Symbol" in out.columns else pd.Series(1, index=out.index)
        keyword_score = out.get("Symbol", pd.Series("", index=out.index)).fillna("").astype(str).map(self._mechanism_keyword_score)
        out["mechanism_tags"] = out.get("Symbol", pd.Series("", index=out.index)).fillna("").astype(str).map(self._mechanism_tags)
        out["target_priority_score"] = (
            self._numeric_series(out, "compound_priority_score").fillna(self._numeric_series(out, "Max_Tox_Prob")).fillna(0) * 0.55
            + (degree / max(float(degree.max()), 1.0)).fillna(0) * 0.25
            + keyword_score.fillna(0) * 0.20
        ).round(4)
        return out.sort_values("target_priority_score", ascending=False)

    def _add_pathway_priority(self, pathway_df: pd.DataFrame) -> pd.DataFrame:
        if pathway_df.empty:
            return pathway_df.copy()
        out = pathway_df.copy()
        name_col = "PathwayName" if "PathwayName" in out.columns else out.columns[-1]
        pathway_names = out[name_col].fillna("").astype(str).str.strip()
        non_empty_names = pathway_names[pathway_names != ""]
        hit_count = pathway_names.map(non_empty_names.value_counts()).fillna(0).astype(int)
        keyword_score = pathway_names.map(self._mechanism_keyword_score)
        out["hit_count"] = hit_count
        out["mechanism_tags"] = pathway_names.map(self._mechanism_tags)
        max_hits = max(float(hit_count.max()) if len(hit_count) else 0.0, 1.0)
        out["pathway_priority_score"] = (
            (hit_count / max_hits).fillna(0) * 0.65
            + keyword_score.fillna(0) * 0.35
        ).round(4)
        return out.sort_values("pathway_priority_score", ascending=False)

    def _add_go_priority(self, go_df: pd.DataFrame) -> pd.DataFrame:
        if go_df.empty:
            return go_df.copy()
        out = go_df.copy()
        term_col = "TERM" if "TERM" in out.columns else ("GOTermName" if "GOTermName" in out.columns else out.columns[-1])
        text = out[term_col].fillna("").astype(str)
        keyword_score = text.map(self._mechanism_keyword_score)
        generic_penalty = text.str.contains("protein binding|binding|cellular process", case=False, regex=True, na=False).astype(float) * 0.35
        out["go_category"] = out.get("Ontology", pd.Series("", index=out.index)).fillna("").astype(str)
        out["go_priority_score"] = (keyword_score - generic_penalty + 0.35).clip(lower=0).round(4)
        return out.sort_values("go_priority_score", ascending=False)

    def _add_disease_priority(self, disease_df: pd.DataFrame) -> pd.DataFrame:
        if disease_df.empty:
            return disease_df.copy()
        out = disease_df.copy()
        name_col = "DiseaseName" if "DiseaseName" in out.columns else out.columns[-1]
        text = out[name_col].fillna("").astype(str)
        out["disease_category"] = text.map(lambda value: "hepatic" if self._mechanism_keyword_score(value) >= 0.5 else "other")
        out["disease_priority_score"] = (text.map(self._mechanism_keyword_score) + 0.15).clip(upper=1).round(4)
        return out.sort_values("disease_priority_score", ascending=False)

    @staticmethod
    def _mechanism_keyword_score(value: str) -> float:
        text = str(value).casefold()
        if not text:
            return 0.0
        score = 0.0
        for keyword in HEPATIC_KEYWORDS:
            if keyword.casefold() in text:
                score += 0.25
        for keyword in ("cyp", "oxidative", "mitochond", "inflamm", "apoptosis", "drug metabolism"):
            if keyword in text:
                score += 0.2
        return min(score, 1.0)

    def _mechanism_tags(self, value: str) -> list[str]:
        text = str(value).casefold()
        tags: list[str] = []
        mapping = {
            "CYP/药物代谢": ("cyp", "drug metabolism", "metabolism"),
            "胆汁酸/胆汁淤积": ("bile", "cholestasis", "胆汁", "胆酸"),
            "氧化应激": ("oxidative", "redox", "氧化"),
            "线粒体": ("mitochond", "线粒体"),
            "炎症": ("inflamm", "炎症"),
            "肝损伤": ("liver", "hepatic", "hepat", "肝"),
        }
        for label, keys in mapping.items():
            if any(key in text for key in keys):
                tags.append(label)
        return tags

    def _build_confidence_profile(
        self,
        livertox_df: pd.DataFrame,
        summary_row: dict[str, Any],
        probabilities: dict[str, float],
    ) -> dict[str, Any]:
        score = 0.9
        reasons: list[str] = []
        if probabilities.get("animal", 0) >= 0.999:
            score -= 0.18
            reasons.append("animal_probability_saturation")
        model_values = [probabilities.get("cell", 0), probabilities.get("animal", 0), probabilities.get("clinical", 0)]
        if max(model_values) - min(model_values) >= 0.55:
            score -= 0.16
            reasons.append("large_model_disagreement")
        valid = self._int_value(summary_row.get("Original_valid_molecule_count"), int(len(livertox_df)))
        total = int(len(livertox_df))
        if total and valid < total:
            score -= min(0.18, (total - valid) / total * 0.3)
            reasons.append("smiles_missing_or_invalid")
        domain = self._build_applicability_domain(livertox_df)
        if domain["domain_label"] == "out_of_domain":
            score -= 0.25
            reasons.append("applicability_domain_outlier")
        elif domain["domain_label"] == "borderline":
            score -= 0.12
            reasons.append("applicability_domain_borderline")
        if not reasons:
            reasons.append("no_major_degradation_rule_triggered")
        score = round(max(0.05, min(score, 0.98)), 4)
        if score >= 0.75:
            level = "high"
        elif score >= 0.45:
            level = "medium"
        elif score >= 0.2:
            level = "low"
        else:
            level = "insufficient"
        return {"confidence_level": level, "confidence_score": score, "confidence_reasons": reasons}

    def _build_applicability_domain(self, livertox_df: pd.DataFrame) -> dict[str, Any]:
        if livertox_df.empty:
            return {"domain_label": "insufficient", "domain_coverage": 0.0, "counts": {}}
        labels = self._domain_label_series(livertox_df)
        counts = labels.value_counts().to_dict()
        coverage = round(float((labels == "in_domain").mean()), 4)
        if coverage >= 0.75:
            label = "in_domain"
        elif coverage >= 0.35:
            label = "borderline"
        elif labels.eq("insufficient").all():
            label = "insufficient"
        else:
            label = "out_of_domain"
        return {"domain_label": label, "domain_coverage": coverage, "counts": counts}

    def _exposure_weighted_risk(self, livertox_df: pd.DataFrame) -> float | None:
        if livertox_df.empty or "Max_Tox_Prob" not in livertox_df.columns:
            return None
        risk = self._numeric_series(livertox_df, "Max_Tox_Prob").fillna(0)
        weights = self._numeric_series(livertox_df, "Bioavailability_Ma").fillna(0.5).clip(0.05, 1.0)
        return round(float((risk * weights).sum() / max(float(weights.sum()), 1e-9)), 4)

    @staticmethod
    def _top_names(df: pd.DataFrame, name_col: str, score_col: str, limit: int) -> list[str]:
        if df.empty or name_col not in df.columns:
            return []
        source = df.copy()
        if score_col in source.columns:
            source = source.sort_values(score_col, ascending=False)
        names = source[name_col].dropna().astype(str).str.strip()
        return [name for name in names.drop_duplicates().head(limit).tolist() if name]

    def _section_file_map(self, query_type: str, config: dict[str, Any] | None = None) -> dict[str, tuple[str, str]]:
        config = config or EXTERNAL_PREDICTION_CONFIG[query_type]
        mapping: dict[str, tuple[str, str]] = {}
        for title, filename in config["overview_files"]:
            mapping[Path(filename).stem] = (title, filename)
        for section_id, (title, filename) in config["tab_files"].items():
            mapping[section_id] = (title, filename)
        return mapping

    @staticmethod
    def _csv_row_count(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            return max(sum(1 for _ in handle) - 1, 0)

    def _manifest_row_count(self, output_dir: Path, filename: str) -> int | None:
        manifest_path = Path(output_dir) / "00_output_file_manifest.csv"
        if not manifest_path.exists():
            return None
        try:
            manifest = self._read_external_csv(manifest_path)
        except PredictionError:
            return None
        if "Rows" not in manifest.columns:
            return None
        names = pd.Series("", index=manifest.index, dtype=object)
        for column in ("FileName", "RelativePath"):
            if column in manifest.columns:
                names = names.mask(names == "", manifest[column].fillna("").astype(str).map(lambda value: Path(value).name))
        matched = manifest[names == Path(filename).name]
        if matched.empty:
            return None
        rows = pd.to_numeric(matched.iloc[0].get("Rows"), errors="coerce")
        if pd.isna(rows):
            return None
        return max(int(rows), 0)

    def _section_row_count(self, output_dir: Path, filename: str) -> int:
        manifest_count = self._manifest_row_count(output_dir, filename)
        if manifest_count is not None:
            return manifest_count
        return self._csv_row_count(Path(output_dir) / filename)

    def _iter_external_csv_chunks(self, path: Path, chunksize: int = EXTERNAL_SECTION_CHUNK_SIZE):
        try:
            yield from pd.read_csv(path, chunksize=chunksize, low_memory=False)
        except UnicodeDecodeError:
            yield from pd.read_csv(path, encoding="gbk", chunksize=chunksize, low_memory=False)

    def _build_section_index(
        self,
        query_type: str,
        output_dir: Path,
        config: dict[str, Any],
        job_id: str | None = None,
    ) -> list[dict[str, Any]]:
        sections = []
        for section_id, (title, filename) in self._section_file_map(query_type, config).items():
            path = output_dir / filename
            row_count = self._section_row_count(output_dir, filename)
            page_api = f"/api/jobs/{job_id}/sections/{section_id}" if job_id else None
            sections.append(
                {
                    "id": section_id,
                    "title": title,
                    "row_count": row_count,
                    "page_api": page_api,
                    "default_sort": self._default_sort_for_section(section_id),
                    "available_filters": ["keyword"],
                    "source_file": filename,
                    "truncated": row_count > EXTERNAL_CSV_SECTION_ROW_LIMIT,
                }
            )
        return sections

    @staticmethod
    def _default_sort_for_section(section_id: str) -> str | None:
        return {
            "toxic_compounds": "compound_priority_score",
            "toxic_targets": "target_priority_score",
            "toxic_pathways": "pathway_priority_score",
            "toxic_go": "go_priority_score",
            "toxic_diseases": "disease_priority_score",
        }.get(section_id)

    def _augment_section_frame(self, section_id: str, df: pd.DataFrame) -> pd.DataFrame:
        if section_id == "toxic_compounds":
            return self._add_compound_priority(self._fill_compound_names_from_cid(df))
        if section_id == "toxic_targets":
            return self._add_target_priority(self._fill_target_identifiers(df), pd.DataFrame())
        if section_id == "toxic_pathways":
            return self._add_pathway_priority(self._fill_target_identifiers(df))
        if section_id == "toxic_go":
            return self._add_go_priority(df)
        if section_id == "toxic_diseases":
            return self._add_disease_priority(self._fill_target_identifiers(df))
        return df

    def _mechanism_section_frame(self, query_type: str, output_dir: Path, section_id: str) -> pd.DataFrame:
        config = EXTERNAL_PREDICTION_CONFIG[query_type]
        title, filename = config["tab_files"][section_id]
        path = Path(output_dir) / filename
        raw = self._read_external_csv(path)

        if section_id == "toxic_targets":
            df = _coalesce_relation_columns(raw)
            if df.empty:
                high_path = Path(output_dir) / config["tab_files"]["toxic_compounds"][1]
                high_toxic = self._fill_compound_names_from_cid(self._read_external_csv(high_path))
                df = self._target_rows_for_high_toxic(high_toxic)
            df = self._fill_target_identifiers(df)
            return self._add_target_priority(df, pd.DataFrame())

        if section_id == "toxic_pathways":
            df = _normalize_relation_output_frame(raw, section_id)
            if df.empty:
                target_df = self._mechanism_section_frame(query_type, output_dir, "toxic_targets")
                df = self._pathway_rows_for_targets(target_df)
            df = self._fill_target_identifiers(df)
            return self._add_pathway_priority(df)

        if section_id == "toxic_go":
            df = _normalize_relation_output_frame(raw, section_id)
            if df.empty:
                target_df = self._mechanism_section_frame(query_type, output_dir, "toxic_targets")
                df = self._go_rows_for_targets(target_df)
            return self._add_go_priority(df)

        raise PredictionError(f"未知结果分区: {section_id}", status_code=404)

    @staticmethod
    def _filter_section_keyword(df: pd.DataFrame, keyword: str | None) -> pd.DataFrame:
        key = (keyword or "").strip()
        if not key:
            return df
        mask = pd.Series(False, index=df.index)
        for column in df.columns:
            mask |= df[column].fillna("").astype(str).str.contains(re.escape(key), case=False, na=False)
        return df[mask]

    @staticmethod
    def _section_page_payload(
        *,
        job_id: str | None,
        section_id: str,
        title: str,
        filename: str,
        row_count: int,
        page: int,
        page_size: int,
        sort_col: str | None,
        order: str,
        keyword: str | None,
        visible: pd.DataFrame,
    ) -> dict[str, Any]:
        visible = _normalize_relation_output_frame(visible, section_id)
        return {
            "ok": True,
            "job_id": job_id,
            "section": {
                "id": section_id,
                "title": title,
                "source_file": filename,
                "row_count": row_count,
                "page": page,
                "page_size": page_size,
                "total_pages": max(1, math.ceil(row_count / page_size)),
                "sort": sort_col,
                "order": order,
                "keyword": keyword or "",
            },
            "headers": [str(column) for column in visible.columns],
            "rows": [
                [HepatotoxicityPredictor._json_safe_value(value) for value in row]
                for row in visible.itertuples(index=False, name=None)
            ],
            "records": [
                {str(key): HepatotoxicityPredictor._json_safe_value(value) for key, value in row.items()}
                for row in visible.to_dict(orient="records")
            ],
        }

    def _should_stream_section_page(self, path: Path) -> bool:
        return path.exists() and path.stat().st_size > EXTERNAL_LARGE_CSV_THRESHOLD_BYTES

    def _get_output_section_page_streamed(
        self,
        *,
        output_dir: Path,
        section_id: str,
        title: str,
        filename: str,
        path: Path,
        page: int,
        page_size: int,
        sort_col: str | None,
        order: str,
        keyword: str | None,
        job_id: str | None,
    ) -> dict[str, Any]:
        start = (page - 1) * page_size
        end = start + page_size
        ascending = order.lower() == "asc"
        manifest_count = self._manifest_row_count(output_dir, filename)
        has_keyword = bool((keyword or "").strip())
        columns: list[str] = []
        selected: list[pd.DataFrame] = []
        candidates = pd.DataFrame()
        matched_count = 0
        sort_available = False

        for chunk in self._iter_external_csv_chunks(path):
            chunk = _normalize_relation_output_frame(chunk, section_id)
            chunk = self._augment_section_frame(section_id, chunk)
            if not columns:
                columns = [str(column) for column in _normalize_relation_output_frame(chunk, section_id).columns]
            chunk = self._filter_section_keyword(chunk, keyword)
            if chunk.empty:
                continue

            if sort_col and sort_col in chunk.columns:
                sort_available = True
                matched_count += len(chunk)
                candidates = pd.concat([candidates, chunk], ignore_index=True)
                candidates = candidates.sort_values(sort_col, ascending=ascending, na_position="last").head(end)
                continue

            next_count = matched_count + len(chunk)
            if next_count > start and matched_count < end:
                chunk_start = max(0, start - matched_count)
                chunk_end = min(len(chunk), end - matched_count)
                selected.append(chunk.iloc[chunk_start:chunk_end])
            matched_count = next_count
            if not has_keyword and manifest_count is not None and matched_count >= end:
                break

        if sort_available:
            row_count = matched_count
            visible = candidates.sort_values(sort_col, ascending=ascending, na_position="last").iloc[start:end]
        else:
            row_count = manifest_count if not has_keyword and manifest_count is not None else matched_count
            visible = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=columns)

        return self._section_page_payload(
            job_id=job_id,
            section_id=section_id,
            title=title,
            filename=filename,
            row_count=int(row_count),
            page=page,
            page_size=page_size,
            sort_col=sort_col if sort_available or not sort_col else None,
            order=order,
            keyword=keyword,
            visible=visible,
        )

    def get_output_section_page(
        self,
        *,
        query_type: str,
        output_dir: Path,
        section_id: str,
        page: int,
        page_size: int,
        sort: str | None = None,
        order: str = "desc",
        keyword: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        mapping = self._section_file_map(query_type)
        if section_id not in mapping:
            raise PredictionError(f"未知结果分区: {section_id}", status_code=404)
        title, filename = mapping[section_id]
        path = Path(output_dir) / filename
        page_size = max(1, min(int(page_size or 50), 500))
        page = max(1, int(page or 1))
        sort_col = sort or self._default_sort_for_section(section_id)
        if section_id in {"toxic_targets", "toxic_pathways", "toxic_go"}:
            df = self._mechanism_section_frame(query_type, Path(output_dir), section_id)
            df = _normalize_relation_output_frame(df, section_id)
            df = self._augment_section_frame(section_id, df)
            df = self._filter_section_keyword(df, keyword)
            if sort_col and sort_col in df.columns:
                ascending = order.lower() == "asc"
                df = df.sort_values(sort_col, ascending=ascending, na_position="last")
            else:
                sort_col = None
            row_count = int(len(df))
            start = (page - 1) * page_size
            end = start + page_size
            visible = df.iloc[start:end].replace({np.nan: None})
            return self._section_page_payload(
                job_id=job_id,
                section_id=section_id,
                title=title,
                filename=filename,
                row_count=row_count,
                page=page,
                page_size=page_size,
                sort_col=sort_col,
                order=order,
                keyword=keyword,
                visible=visible,
            )
        if section_id == "toxic_diseases":
            df = self._read_external_csv(path)
            target_filename = EXTERNAL_PREDICTION_CONFIG[query_type]["tab_files"]["toxic_targets"][1]
            target_path = Path(output_dir) / target_filename
            if target_path.exists():
                target_df = self._mechanism_section_frame(query_type, Path(output_dir), "toxic_targets")
                df = self._target_disease_frame_from_targets(target_df, df)
            df = _normalize_relation_output_frame(df, section_id)
            df = self._augment_section_frame(section_id, df)
            df = df[_valid_toxic_disease_mask(df)].copy()
            df = self._filter_section_keyword(df, keyword)
            if sort_col and sort_col in df.columns:
                ascending = order.lower() == "asc"
                df = df.sort_values(sort_col, ascending=ascending, na_position="last")
            else:
                sort_col = None
            row_count = int(len(df))
            start = (page - 1) * page_size
            end = start + page_size
            visible = df.iloc[start:end].replace({np.nan: None})
            return self._section_page_payload(
                job_id=job_id,
                section_id=section_id,
                title=title,
                filename=filename,
                row_count=row_count,
                page=page,
                page_size=page_size,
                sort_col=sort_col,
                order=order,
                keyword=keyword,
                visible=visible,
            )
        if self._should_stream_section_page(path):
            return self._get_output_section_page_streamed(
                output_dir=Path(output_dir),
                section_id=section_id,
                title=title,
                filename=filename,
                path=path,
                page=page,
                page_size=page_size,
                sort_col=sort_col,
                order=order,
                keyword=keyword,
                job_id=job_id,
            )
        df = self._read_external_csv(path)
        df = _normalize_relation_output_frame(df, section_id)
        df = self._augment_section_frame(section_id, df)
        df = self._filter_section_keyword(df, keyword)
        if sort_col and sort_col in df.columns:
            ascending = order.lower() == "asc"
            df = df.sort_values(sort_col, ascending=ascending, na_position="last")
        else:
            sort_col = None
        row_count = int(len(df))
        start = (page - 1) * page_size
        end = start + page_size
        visible = df.iloc[start:end].replace({np.nan: None})
        return self._section_page_payload(
            job_id=job_id,
            section_id=section_id,
            title=title,
            filename=filename,
            row_count=row_count,
            page=page,
            page_size=page_size,
            sort_col=sort_col,
            order=order,
            keyword=keyword,
            visible=visible,
        )

    @staticmethod
    def _chart_spec(
        chart_id: str,
        title: str,
        chart_type: str,
        data: list[dict[str, Any]],
        encoding: dict[str, str],
        *,
        description: str = "",
        source_section: str = "summary",
        filters: dict[str, Any] | None = None,
        click_action: dict[str, Any] | None = None,
        exportable: bool = True,
    ) -> dict[str, Any]:
        return {
            "id": chart_id,
            "title": title,
            "type": chart_type,
            "description": description,
            "data": data,
            "encoding": encoding,
            "filters": filters or {},
            "source_section": source_section,
            "click_action": click_action or {},
            "exportable": exportable,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def build_prediction_visuals_from_frames(
        self,
        summary: dict[str, Any],
        probabilities: dict[str, float],
        high_toxic_df: pd.DataFrame,
        target_df: pd.DataFrame,
        pathway_df: pd.DataFrame,
        go_df: pd.DataFrame,
        disease_df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        risk_data = [
            {"model": "Cell", "score_type": "raw", "probability": probabilities.get("cell", 0)},
            {"model": "Animal", "score_type": "raw", "probability": probabilities.get("animal", 0)},
            {"model": "Clinical", "score_type": "raw", "probability": probabilities.get("clinical", 0)},
        ]
        specs.append(self._chart_spec("risk_model_bar", "三模型肝毒性概率", "bar", risk_data, {"x": "model", "y": "probability", "series": "score_type"}, description="显示三个模型原始概率；校准模型缺失时不展示校准概率。", source_section="final"))
        specs.append(
            self._chart_spec(
                "prediction_funnel",
                "预测流程收敛",
                "funnel",
                [
                    {"stage": "总成分", "count": summary.get("total_compounds", 0)},
                    {"stage": "有效SMILES", "count": summary.get("valid_smiles_count", 0)},
                    {"stage": "入血成分", "count": summary.get("intoblood_count", 0)},
                    {"stage": "有效吸收", "count": summary.get("absorbed_count", 0)},
                    {"stage": "高危肝毒", "count": summary.get("high_toxic_count", 0)},
                ],
                {"x": "stage", "y": "count"},
                source_section="overview",
            )
        )
        if not high_toxic_df.empty:
            top_compounds = self._records_from_external_df(high_toxic_df.head(20), None)
            specs.append(self._chart_spec("top_compounds_bar", "高危成分 TOP20", "bar", top_compounds, {"x": "ChemicalName", "y": "compound_priority_score"}, source_section="toxic_compounds", click_action={"filter_section": "toxic_compounds", "field": "ChemicalName"}))
            source_data = self._source_herb_contribution(high_toxic_df)
            specs.append(self._chart_spec("source_herb_contribution_bar", "来源中药风险贡献", "bar", source_data, {"x": "herb", "y": "high_toxic_count", "series": "metric"}, source_section="toxic_compounds"))
            class_data = self._distribution_records(high_toxic_df, "Class", 10)
            specs.append(self._chart_spec("class_distribution", "高危成分类别分布", "pie", class_data, {"name": "name", "value": "count"}, source_section="toxic_compounds"))
        if not target_df.empty:
            specs.append(self._chart_spec("target_network_graph", "成分-靶标-通路网络", "graph", self._target_network_records(high_toxic_df, target_df, pathway_df), {"nodes": "nodes", "edges": "edges"}, source_section="toxic_targets", click_action={"filter_section": "toxic_targets", "field": "name"}))
        if not pathway_df.empty:
            specs.append(self._chart_spec("pathway_enrichment_bar", "通路机制优先级", "bar", self._records_from_external_df(pathway_df.head(20), None), {"x": "PathwayName", "y": "pathway_priority_score", "size": "hit_count"}, source_section="toxic_pathways"))
        if not go_df.empty:
            term_col = "TERM" if "TERM" in go_df.columns else ("GOTermName" if "GOTermName" in go_df.columns else go_df.columns[-1])
            go_data = self._records_from_external_df(go_df.head(30).rename(columns={term_col: "TERM"}), None)
            specs.append(self._chart_spec("go_mechanism_treemap", "GO机制分类", "treemap", go_data, {"name": "TERM", "value": "go_priority_score", "category": "go_category"}, source_section="toxic_go"))
        if not disease_df.empty:
            specs.append(self._chart_spec("disease_top_bar", "肝病相关疾病 TOP20", "bar", self._records_from_external_df(disease_df.head(20), None), {"x": "DiseaseName", "y": "disease_priority_score"}, source_section="toxic_diseases"))
        return specs

    @staticmethod
    def _distribution_records(df: pd.DataFrame, column: str, limit: int) -> list[dict[str, Any]]:
        if df.empty or column not in df.columns:
            return []
        counts = df[column].fillna("未分类").astype(str).value_counts()
        top = counts.head(limit)
        records = [{"name": str(name), "count": int(count)} for name, count in top.items()]
        other = int(counts.iloc[limit:].sum()) if len(counts) > limit else 0
        if other:
            records.append({"name": "Other", "count": other})
        return records

    @staticmethod
    def _database_source_rows() -> list[dict[str, Any]]:
        return [
            {"entity": "方剂", "source": "中国药典2025版", "count": 1593},
            {"entity": "方剂", "source": "中医世家", "count": 31170},
            {"entity": "中药", "source": "中国药典2025版", "count": 1599},
            {"entity": "中药", "source": "中医世家", "count": 14272},
            {"entity": "中药", "source": "BATMAN-TCM", "count": 2027},
            {"entity": "中药", "source": "DCABM-TCM", "count": 192},
            {"entity": "中药", "source": "TCMSP", "count": 502},
            {"entity": "化学成分", "source": "BATMAN-TCM", "count": 24145},
            {"entity": "化学成分", "source": "DCABM-TCM", "count": 2426},
            {"entity": "化学成分", "source": "TCMSP", "count": 13225},
            {"entity": "成分类别", "source": "NPClassifier", "count": 1227},
            {"entity": "生物靶标", "source": "CTD", "count": 41944},
            {"entity": "生物靶标", "source": "iTCM", "count": 3978},
            {"entity": "GO术语", "source": "CTD", "count": 11976},
            {"entity": "GO术语", "source": "GO", "count": 17607},
            {"entity": "信号通路", "source": "KEGG", "count": 370},
            {"entity": "疾病信息", "source": "CTD", "count": 7281},
            {"entity": "入血化合物", "source": "DCABM-TCM", "count": 2426},
        ]

    @classmethod
    def _database_source_total_records(cls) -> list[dict[str, Any]]:
        totals: dict[tuple[str, str], int] = {}
        for row in cls._database_source_rows():
            key = (str(row["source"]), str(row["entity"]))
            totals[key] = totals.get(key, 0) + int(row["count"])
        return [
            {"source": source, "entity": entity, "count": count}
            for (source, entity), count in sorted(totals.items(), key=lambda item: (item[0][0], item[0][1]))
        ]

    @staticmethod
    def _database_source_cleaning_records() -> list[dict[str, Any]]:
        rows = [
            ("方剂", 1593 + 31170, 117, 32646),
            ("中药", 1599 + 14272 + 2027 + 192 + 502, 24591, 16119),
            ("化学成分", 24145 + 2426 + 13225, 2261, 37535),
            ("成分类别", 1227, 0, 1227),
            ("生物靶标", 41944 + 3978, 9923, 35999),
            ("GO术语", 11976 + 17607, 11951, 17632),
            ("信号通路", 370, 18, 352),
            ("疾病信息", 7281, 0, 7281),
            ("入血化合物", 2426, 699, 1727),
        ]
        records: list[dict[str, Any]] = []
        for entity, raw_total, deduplicated, final_count in rows:
            records.extend(
                [
                    {"entity": entity, "metric": "原始数据量", "count": raw_total},
                    {"entity": entity, "metric": "清洗去重", "count": deduplicated},
                    {"entity": entity, "metric": "最终结果", "count": final_count},
                ]
            )
        return records

    @staticmethod
    def _source_herb_contribution(high_toxic_df: pd.DataFrame) -> list[dict[str, Any]]:
        if high_toxic_df.empty:
            return []
        rows: list[dict[str, Any]] = []
        source_col = "Source_Herbs" if "Source_Herbs" in high_toxic_df.columns else "Herb.Chinese.name"
        if source_col not in high_toxic_df.columns:
            return []
        for _, row in high_toxic_df.iterrows():
            herbs = [part.strip() for part in re.split(r"[;；,，、]+", str(row.get(source_col) or "")) if part.strip()]
            risk = _native(row.get("Max_Tox_Prob")) or 0
            for herb in herbs:
                rows.append({"herb": herb, "risk": float(risk)})
        if not rows:
            return []
        frame = pd.DataFrame(rows)
        grouped = frame.groupby("herb").agg(high_toxic_count=("risk", "size"), mean_risk=("risk", "mean")).reset_index()
        grouped = grouped.sort_values(["high_toxic_count", "mean_risk"], ascending=False).head(20)
        return [
            {"herb": str(row.herb), "high_toxic_count": int(row.high_toxic_count), "mean_risk": round(float(row.mean_risk), 4), "metric": "count"}
            for row in grouped.itertuples(index=False)
        ]

    def _target_network_records(self, high_toxic_df: pd.DataFrame, target_df: pd.DataFrame, pathway_df: pd.DataFrame) -> list[dict[str, Any]]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        compound_names = self._top_names(high_toxic_df, "ChemicalName", "compound_priority_score", 20)
        for name in compound_names:
            nodes[f"compound:{name}"] = {"id": f"compound:{name}", "name": name, "type": "compound", "score": 1, "size": 18, "category": "compound"}
        target_subset = target_df.copy()
        if compound_names and "ChemicalName" in target_subset.columns:
            target_subset = target_subset[target_subset["ChemicalName"].isin(compound_names)]
        target_subset = target_subset.head(120)
        for row in target_subset.to_dict(orient="records"):
            compound = str(row.get("ChemicalName") or "")
            symbol = str(row.get("Symbol") or "")
            if not symbol:
                continue
            target_id = f"target:{symbol}"
            nodes[target_id] = {"id": target_id, "name": symbol, "type": "target", "score": row.get("target_priority_score", 0.5), "size": 14, "category": "target"}
            if compound:
                nodes.setdefault(f"compound:{compound}", {"id": f"compound:{compound}", "name": compound, "type": "compound", "score": row.get("Max_Tox_Prob", 0.5), "size": 16, "category": "compound"})
                edges.append({"source": f"compound:{compound}", "target": target_id, "relation": "targets", "weight": row.get("target_priority_score", 0.5), "evidence": "compound_target"})
        if not pathway_df.empty and "Symbol" in pathway_df.columns:
            pathway_subset = pathway_df[pathway_df["Symbol"].isin([node["name"] for node in nodes.values() if node["type"] == "target"])].head(120)
            for row in pathway_subset.to_dict(orient="records"):
                symbol = str(row.get("Symbol") or "")
                pathway = str(row.get("PathwayName") or row.get("Pathwayid") or "")
                if not symbol or not pathway:
                    continue
                pathway_id = f"pathway:{pathway}"
                nodes[pathway_id] = {"id": pathway_id, "name": pathway, "type": "pathway", "score": row.get("pathway_priority_score", 0.5), "size": 12, "category": "pathway"}
                edges.append({"source": f"target:{symbol}", "target": pathway_id, "relation": "in_pathway", "weight": row.get("pathway_priority_score", 0.5), "evidence": "target_pathway"})
        return [{"nodes": list(nodes.values()), "edges": edges}]

    def build_database_visuals(self) -> dict[str, Any]:
        data = self._load_data()
        stats = self.database_stats().get("stats", {})
        overview_data = [
            {"entity": "方剂", "count": stats.get("formulas", 0)},
            {"entity": "中药", "count": stats.get("herbs", 0)},
            {"entity": "化学成分", "count": stats.get("compounds", 0)},
            {"entity": "化学类别", "count": stats.get("classes", 0)},
            {"entity": "化学超类", "count": stats.get("superclasses", 0)},
            {"entity": "化学通路", "count": stats.get("chemical_pathways", 0)},
            {"entity": "靶标", "count": stats.get("targets", 0)},
            {"entity": "信号通路", "count": stats.get("signal_pathways", 0)},
            {"entity": "GO术语", "count": stats.get("go_terms", 0)},
            {"entity": "疾病", "count": stats.get("diseases", 0)},
        ]
        compound_class = data["compound_class"]
        source_records = self._database_source_total_records()
        cleaning_records = self._database_source_cleaning_records()
        specs = [
            self._chart_spec(
                "database_overview_bar",
                "数据库信息总览",
                "bar",
                overview_data,
                {"x": "entity", "y": "count"},
                description="方剂、中药、化学成分、分类、靶标、通路、GO与疾病等核心数据库规模。",
                source_section="database",
            ),
            self._chart_spec("compound_class_top", "化学成分类别", "bar", self._distribution_records(compound_class, "Class", 15), {"x": "name", "y": "count"}, source_section="database", click_action={"input_type": "class", "field": "name"}),
            self._chart_spec(
                "data_source_total_bar",
                "数据库来源总览",
                "bar",
                source_records,
                {"x": "source", "y": "count", "series": "entity"},
                description="按来源数据库、药典或文献统计各数据层级的原始来源数量。",
                source_section="database_source",
                filters={"log_y": True, "full_width": True},
            ),
            self._chart_spec(
                "entity_source_grouped_bar",
                "各类数据来源总览",
                "bar",
                list(self._database_source_rows()),
                {"x": "entity", "y": "count", "series": "source"},
                description="展示方剂、中药、化学成分、靶标、GO、通路、疾病和入血化合物的来源构成。",
                source_section="database_source",
                filters={"log_y": True, "full_width": True},
            ),
            self._chart_spec(
                "source_cleaning_result_bar",
                "数据清洗统计",
                "bar",
                cleaning_records,
                {"x": "entity", "y": "count", "series": "metric"},
                description="按数据层级对比原始数据量、清洗去重数量和最终入库结果。",
                source_section="database_source",
                filters={"full_width": True},
            ),
        ]
        manifest = self._load_manifest(self.paths.root / "data_manifest.json")
        return {"ok": True, "chart_specs": specs, "stats": stats, "data_manifest": manifest}

    def build_knowledge_visuals(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        summary = result.get("summary", {})
        trace = result.get("trace_details", {})
        entity_count = [
            {"entity": "关联成分", "count": summary.get("matched_compounds", 0)},
            {"entity": "关联靶标", "count": summary.get("matched_targets", 0)},
            {"entity": "上游中药", "count": summary.get("matched_herbs", 0)},
            {"entity": "覆盖方剂", "count": summary.get("matched_formulas", 0)},
        ]
        specs = [
            self._chart_spec("knowledge_entity_count_bar", "知识图谱实体数量", "bar", entity_count, {"x": "entity", "y": "count"}, source_section="knowledge_overview")
        ]
        compounds = trace.get("compound_classes", [])[:40]
        targets = trace.get("compound_targets", [])[:80]
        herbs = trace.get("herb_compounds", [])[:80]
        formulas = trace.get("formula_herbs", [])[:80]
        specs.append(self._chart_spec("knowledge_network_graph", "中心实体关联网络", "graph", self._knowledge_network_records(result.get("query", []), compounds, targets, herbs, formulas), {"nodes": "nodes", "edges": "edges"}, source_section="knowledge_overview"))
        if herbs:
            herb_frame = pd.DataFrame(herbs)
            specs.append(self._chart_spec("upstream_herb_top_bar", "上游中药 TOP20", "bar", self._distribution_records(herb_frame, "Herb.Chinese.name", 20), {"x": "name", "y": "count"}, source_section="formula_herbs", click_action={"input_type": "herb", "field": "name"}))
        if formulas:
            formula_frame = pd.DataFrame(formulas)
            specs.append(self._chart_spec("formula_coverage_top_bar", "方剂覆盖 TOP20", "bar", self._distribution_records(formula_frame, "Formula.Chinese.name", 20), {"x": "name", "y": "count"}, source_section="formula_herbs"))
        if compounds:
            compound_frame = pd.DataFrame(compounds)
            specs.append(self._chart_spec("knowledge_compound_class_bar", "关联成分类别", "bar", self._distribution_records(compound_frame, "Class", 15), {"x": "name", "y": "count"}, source_section="compound_classes"))
        return specs

    @staticmethod
    def _knowledge_network_records(
        query: list[str],
        compounds: list[dict[str, Any]],
        targets: list[dict[str, Any]],
        herbs: list[dict[str, Any]],
        formulas: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        center = "、".join(query) if query else "query"
        nodes: dict[str, dict[str, Any]] = {
            "query": {"id": "query", "name": center, "type": "query", "score": 1, "size": 24, "category": "query"}
        }
        edges: list[dict[str, Any]] = []
        for row in compounds[:30]:
            name = str(row.get("ChemicalName") or row.get("CID") or "")
            if not name:
                continue
            node_id = f"compound:{name}"
            nodes[node_id] = {"id": node_id, "name": name, "type": "compound", "score": 0.8, "size": 15, "category": "compound"}
            edges.append({"source": "query", "target": node_id, "relation": "matches", "weight": 0.8, "evidence": "compound_class"})
        for row in targets[:60]:
            compound = str(row.get("ChemicalName") or "")
            symbol = str(row.get("Symbol") or "")
            if not symbol:
                continue
            node_id = f"target:{symbol}"
            nodes[node_id] = {"id": node_id, "name": symbol, "type": "target", "score": 0.65, "size": 13, "category": "target"}
            source = f"compound:{compound}" if compound and f"compound:{compound}" in nodes else "query"
            edges.append({"source": source, "target": node_id, "relation": "targets", "weight": 0.65, "evidence": "compound_target"})
        for row in herbs[:40]:
            herb = str(row.get("Herb.Chinese.name") or "")
            compound = str(row.get("ChemicalName") or "")
            if not herb:
                continue
            node_id = f"herb:{herb}"
            nodes[node_id] = {"id": node_id, "name": herb, "type": "herb", "score": 0.55, "size": 12, "category": "herb"}
            target = f"compound:{compound}" if compound and f"compound:{compound}" in nodes else "query"
            edges.append({"source": node_id, "target": target, "relation": "contains", "weight": 0.55, "evidence": "herb_compound"})
        for row in formulas[:40]:
            formula = str(row.get("Formula.Chinese.name") or "")
            herb = str(row.get("Herb.Chinese.name") or "")
            if not formula:
                continue
            node_id = f"formula:{formula}"
            nodes[node_id] = {"id": node_id, "name": formula, "type": "formula", "score": 0.45, "size": 10, "category": "formula"}
            target = f"herb:{herb}" if herb and f"herb:{herb}" in nodes else "query"
            edges.append({"source": node_id, "target": target, "relation": "uses", "weight": 0.45, "evidence": "formula_herb"})
        return [{"nodes": list(nodes.values()), "edges": edges}]

    def search_knowledge(self, kind: str, text: str | Iterable[str]) -> dict[str, Any]:
        items = split_items(text)
        if not items:
            raise PredictionError("请输入检索关键词。")
        data = self._load_data()
        kind = kind.lower()

        compounds = data["compound_class"]
        targets = data["compound_target"]
        pathways = data["target_pathway"]
        target_go = data["target_go"]
        go_term = data["go_term"]
        herb_compound = data["herb_compound"]
        formula_herb = data["formula_herb"]
        max_symbols = 1000
        max_compounds = 3000

        if kind == "class":
            mask = (
                _contains_any(compounds["Class"], items)
                | _contains_any(compounds["Superclass"], items)
                | _contains_any(compounds["Pathway"], items)
            )
            matched_compounds = compounds[mask].copy()
            matched_symbols_all = self._targets_for_compounds(matched_compounds.head(max_compounds))
            matched_symbols = matched_symbols_all[:max_symbols]
        elif kind == "target":
            item_norm = {item.strip().lower() for item in items}
            pathway_symbols = pathways["Symbol"].fillna("").astype(str)
            entrez_ids = pathways["ENTREZID"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
            pathway_target_mask = (
                pathway_symbols.str.lower().isin(item_norm)
                | entrez_ids.str.lower().isin(item_norm)
            )
            matched_target_pathways = pathways[pathway_target_mask].copy()
            if matched_target_pathways.empty:
                matched_target_pathways = pathways[_contains_any(pathway_symbols, items)].copy()
            if matched_target_pathways.empty:
                matched_targets = targets[_contains_any(targets["Symbol"], items)].copy()
            else:
                target_symbols = matched_target_pathways["Symbol"].dropna().astype(str).unique().tolist()
                matched_targets = targets[targets["Symbol"].isin(target_symbols)].copy()
            names = matched_targets["ChemicalName"].dropna().unique().tolist()
            matched_compounds = compounds[compounds["ChemicalName"].isin(names)].copy()
            matched_symbols_all = sorted(matched_targets["Symbol"].dropna().astype(str).unique().tolist())
            matched_symbols = matched_symbols_all[:max_symbols]
        elif kind == "pathway":
            item_norm = {item.strip().lower() for item in items}
            pathway_ids = pathways["Pathwayid"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
            mask = _contains_any(pathways["PathwayName"], items) | pathway_ids.str.lower().isin(item_norm)
            matched_pathways = pathways[mask].copy()
            matched_symbols_all = sorted(matched_pathways["Symbol"].dropna().astype(str).unique().tolist())
            matched_symbols = matched_symbols_all[:max_symbols]
            names = targets[targets["Symbol"].isin(matched_symbols)]["ChemicalName"].dropna().unique().tolist()
            matched_compounds = compounds[compounds["ChemicalName"].isin(names)].copy()
        elif kind == "go":
            item_norm = {item.strip().lower() for item in items}
            go_ids = go_term["GOID"].fillna("").astype(str)
            mask = _contains_any(go_term["TERM"], items) | go_ids.str.lower().isin(item_norm)
            matched_go = go_term[mask].copy()
            go_ids = matched_go["GOID"].dropna().astype(str).unique().tolist()[:500]
            matched_symbols_all = sorted(target_go[target_go["GOID"].astype(str).isin(go_ids)]["Symbol"].dropna().astype(str).unique().tolist())
            matched_symbols = matched_symbols_all[:max_symbols]
            names = targets[targets["Symbol"].isin(matched_symbols)]["ChemicalName"].dropna().unique().tolist()
            matched_compounds = compounds[compounds["ChemicalName"].isin(names)].copy()
        else:
            raise PredictionError(f"暂不支持该知识检索维度: {kind}")

        matched_compounds = matched_compounds.head(max_compounds).copy()
        cids = pd.to_numeric(matched_compounds.get("CID", pd.Series(dtype=float)), errors="coerce").dropna().unique().tolist()
        herb_rows = herb_compound[herb_compound["CID_num"].isin(cids)].copy()
        herbs = sorted(herb_rows["Herb.Chinese.name"].dropna().astype(str).unique().tolist())
        formula_rows = formula_herb[formula_herb["Herb.Chinese.name"].isin(herbs)].copy()
        formulas = sorted(formula_rows["Formula.Chinese.name"].dropna().astype(str).unique().tolist())

        matched_pathways = pathways[pathways["Symbol"].isin(matched_symbols)].copy()
        matched_go = target_go[target_go["Symbol"].isin(matched_symbols)].merge(go_term, on="GOID", how="left").copy()
        matched_diseases = self._disease_rows_for_symbols(matched_symbols, data, limit=300)
        chem_disease_rows = self._chem_disease_rows_for_chemicals(matched_compounds, data, limit=1000)
        chem_go_rows = self._chem_go_rows_for_chemicals(matched_compounds, data, limit=1000)
        chem_pathway_rows = self._chem_pathway_rows_for_chemicals(matched_compounds, data, limit=1000)
        matched_target_rows = targets[
            targets["ChemicalName"].isin(matched_compounds["ChemicalName"].dropna().astype(str).unique().tolist())
            & targets["Symbol"].isin(matched_symbols)
        ].copy()

        rows = [
            ["检索关键词", "、".join(items)],
            ["匹配化学成分数", len(matched_compounds)],
            ["关联靶标数", len(set(matched_symbols))],
            ["上游中药数", len(herbs)],
            ["覆盖方剂数", len(formulas)],
            ["高频化学分类", self._join_counts(_top_counts(matched_compounds.get("Class", pd.Series(dtype=str))))],
            ["主要靶标", "、".join(matched_symbols[:20])],
            ["主要通路", self._join_counts(_top_counts(matched_pathways.get("PathwayName", pd.Series(dtype=str))))],
            ["主要GO术语", self._join_counts(_top_counts(matched_go.get("TERM", pd.Series(dtype=str))))],
            ["代表中药", "、".join(herbs[:20])],
            ["代表方剂", "、".join(formulas[:20])],
        ]
        response = {
            "ok": True,
            "query_type": kind,
            "query": items,
            "summary": {
                "matched_compounds": int(len(matched_compounds)),
                "matched_targets": int(len(set(matched_symbols))),
                "matched_herbs": int(len(herbs)),
                "matched_formulas": int(len(formulas)),
            },
            "rows": rows,
            "compounds": _records(matched_compounds, None),
            "targets": matched_symbols[:None],
            "pathways": _records(matched_pathways.drop_duplicates(), None),
            "go_terms": _records(matched_go.drop_duplicates(), None),
            "herbs": herbs[:None],
            "formulas": formulas[:None],
            "trace_details": {
                "formula_herbs": _records(formula_rows[["Formula.Chinese.name", "Herb.Chinese.name"]].drop_duplicates(), 300),
                "herb_compounds": _records(herb_rows[["Herb.Chinese.name", "Herb.Pinyin.name", "CID", "ChemicalName"]].drop_duplicates(), 300),
                "compound_classes": _balanced_records(
                    matched_compounds[[col for col in ["CID", "ChemicalName", "Class", "Superclass", "Pathway"] if col in matched_compounds.columns]]
                    .drop_duplicates(),
                    ["ChemicalName", "CID"],
                    1000,
                ),
                "compound_targets": _balanced_records(self._with_entrez(matched_target_rows).drop_duplicates(), ["ChemicalName"], 1000),
                "chem_diseases": _balanced_records(
                    self._select_columns(chem_disease_rows, ["ChemicalName", "ChemicalID", "DiseaseName", "DiseaseID"]),
                    ["ChemicalName"],
                    1000,
                ),
                "chem_go": _balanced_records(
                    self._select_columns(chem_go_rows, ["ChemicalName", "ChemicalID", "Ontology", "GOTermID", "GOTermName", "PValue", "CorrectedPValue"]),
                    ["ChemicalName"],
                    1000,
                ),
                "chem_pathways": _balanced_records(
                    self._select_columns(chem_pathway_rows, ["ChemicalName", "ChemicalID", "PathwayID", "PathwayName", "PValue", "CorrectedPValue"]),
                    ["ChemicalName"],
                    1000,
                ),
                "target_go": _balanced_records(
                    self._select_columns(matched_go.drop_duplicates(), ["Symbol", "GOID", "Ontology", "TERM"]),
                    ["Symbol"],
                    1000,
                ),
                "target_pathways": _balanced_records(
                    self._select_columns(matched_pathways.drop_duplicates(), ["ENTREZID", "Symbol", "Pathwayid", "PathwayName"]),
                    ["Symbol"],
                    1000,
                ),
                "target_diseases": _balanced_records(
                    self._select_columns(self._with_entrez(matched_diseases), ["ENTREZID", "Symbol", "DiseaseName", "DiseaseID"]),
                    ["Symbol"],
                    1000,
                ),
            },
        }
        response["chart_specs"] = self.build_knowledge_visuals(response)
        return response

    def _run_prediction(
        self,
        query_type: str,
        items: list[str],
        chemicals: pd.DataFrame,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if chemicals.empty:
            raise PredictionError("未在本地数据库中找到对应化学成分。")

        self._warnings = []
        chemicals = self._normalize_chemical_frame(chemicals)
        intoblood = self._predict_intoblood(chemicals)
        evaluated = self._evaluate_toxicity(intoblood)
        return self._build_prediction_response(query_type, items, evaluated, context)

    def _load_data(self, refresh: bool = False) -> dict[str, pd.DataFrame]:
        if self._data is not None and not refresh:
            return self._data
        with self._data_lock:
            if self._data is not None and not refresh:
                return self._data
            formula_herb = _read_csv(self.paths.formula_herb)
            herb_compound = _read_csv(self.paths.herb_compound)
            compound_class = _read_csv(self.paths.compound_class)
            compound_target = _read_csv(self.paths.compound_target)
            target_pathway = _read_csv(self.paths.target_pathway)
            target_go = _read_csv(self.paths.target_go)
            go_term = _read_csv(self.paths.go_term)
            intoblood_reference = _read_csv(self.paths.intoblood_reference, low_memory=False)
            food_medicine_homology = _read_csv(self.paths.food_medicine_homology)

            formula_herb = _copy_column_aliases(formula_herb)
            herb_compound = _copy_column_aliases(herb_compound)
            compound_class = _copy_column_aliases(compound_class)
            compound_target = _copy_column_aliases(compound_target)
            target_pathway = _copy_column_aliases(target_pathway)
            target_go = _copy_column_aliases(target_go)
            go_term = _copy_column_aliases(go_term)
            intoblood_reference = _copy_column_aliases(intoblood_reference)
            food_medicine_homology = _copy_column_aliases(food_medicine_homology)

            for df in (herb_compound, compound_class):
                if "CID" in df.columns:
                    df["CID_num"] = pd.to_numeric(df["CID"], errors="coerce")

            self._data = {
                "formula_herb": formula_herb,
                "herb_compound": herb_compound,
                "compound_class": compound_class,
                "compound_target": compound_target,
                "target_pathway": target_pathway,
                "target_go": target_go,
                "go_term": go_term,
                "intoblood_reference": intoblood_reference,
                "food_medicine_homology": food_medicine_homology,
            }
            return self._data

    def _resolve_formula(self, items: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
        data = self._load_data()
        formula_herb = data["formula_herb"]
        exact = formula_herb[formula_herb["Formula.Chinese.name"].isin(items)].copy()
        if exact.empty:
            exact = formula_herb[_contains_any(formula_herb["Formula.Chinese.name"], items)].copy()
        if exact.empty:
            exact = formula_herb[_contains_any(formula_herb["Formula.Chinese.name"], self._relaxed_formula_terms(items))].copy()
        matched_formulas = sorted(exact["Formula.Chinese.name"].dropna().astype(str).unique().tolist())
        missing = [
            item
            for item in items
            if not any(
                item == hit or item in hit or any(term in hit for term in self._relaxed_formula_terms([item]))
                for hit in matched_formulas
            )
        ]
        herbs = sorted(exact["Herb.Chinese.name"].dropna().astype(str).unique().tolist())
        chemicals = self._chemicals_from_herbs(herbs)
        chemicals["Source_Formulas"] = "、".join(matched_formulas)
        context = {
            "matched_formulas": matched_formulas,
            "missing": missing,
            "herbs": herbs,
            "related_formulas": matched_formulas,
        }
        return chemicals, context

    def _resolve_herb(self, items: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
        data = self._load_data()
        herb_compound = data["herb_compound"]
        formula_herb = data["formula_herb"]
        chinese_names = herb_compound["Herb.Chinese.name"].fillna("").astype(str)
        pinyin_names = herb_compound.get("Herb.Pinyin.name", pd.Series("", index=herb_compound.index)).fillna("").astype(str)
        item_norm = {item.strip().lower() for item in items}

        exact_mask = chinese_names.isin(items) | pinyin_names.str.lower().isin(item_norm)
        matched_rows = herb_compound[exact_mask].copy()
        if matched_rows.empty:
            fuzzy_mask = _contains_any(chinese_names, items) | _contains_any(pinyin_names, items)
            matched_rows = herb_compound[fuzzy_mask].copy()

        herbs = sorted(
            matched_rows["Herb.Chinese.name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        matched_pinyin = (
            matched_rows["Herb.Pinyin.name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
            if "Herb.Pinyin.name" in matched_rows.columns
            else []
        )

        def herb_item_matched(item: str) -> bool:
            item_clean = item.strip()
            item_lower = item_clean.lower()
            return any(item_clean == hit or item_clean in hit for hit in herbs) or any(
                item_lower == pinyin.lower() or item_lower in pinyin.lower()
                for pinyin in matched_pinyin
            )
        related_formulae = sorted(
            formula_herb.loc[formula_herb["Herb.Chinese.name"].isin(herbs), "Formula.Chinese.name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        chemicals = self._chemicals_from_herbs(herbs)
        context = {
            "matched_herbs": herbs,
            "missing": [item for item in items if not herb_item_matched(item)],
            "herbs": herbs,
            "related_formulas": related_formulae,
        }
        return chemicals, context

    def _resolve_compound(self, items: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
        data = self._load_data()
        compounds = data["compound_class"]
        herb_compound = data["herb_compound"]
        formula_herb = data["formula_herb"]

        cid_items: list[int] = []
        smiles_items: list[str] = []
        name_items: list[str] = []
        for item in items:
            item_clean = item.strip()
            cid_match = re.fullmatch(r"(?i)(?:cid[\s:_-]*)?(\d+)(?:\.0+)?", item_clean)
            if cid_match:
                cid_items.append(int(cid_match.group(1)))
            elif Chem.MolFromSmiles(item_clean) is not None:
                smiles_items.append(item_clean)
            else:
                name_items.append(item_clean)

        frames: list[pd.DataFrame] = []
        if cid_items:
            frames.append(compounds[compounds["CID_num"].isin(cid_items)].copy())
        if name_items:
            exact = compounds[compounds["ChemicalName"].fillna("").str.lower().isin([x.lower() for x in name_items])].copy()
            fuzzy = compounds[_contains_any(compounds["ChemicalName"], name_items)].copy()
            frames.extend([exact, fuzzy])
        for smiles in smiles_items:
            matched = compounds[compounds["Smiles"].fillna("").astype(str) == smiles].copy()
            if matched.empty:
                matched = pd.DataFrame(
                    [
                        {
                            "CID": None,
                            "CID_num": np.nan,
                            "ChemicalName": "输入SMILES",
                            "Smiles": smiles,
                            "Class": None,
                            "Superclass": None,
                            "Pathway": None,
                            "Is_glycoside": None,
                        }
                    ]
                )
            frames.append(matched)

        if not frames:
            raise PredictionError("未识别到有效的 CID、SMILES 或化学成分名称。")

        chemicals = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["CID_num", "ChemicalName", "Smiles"])
        cids = pd.to_numeric(chemicals["CID_num"], errors="coerce").dropna().unique().tolist()
        herb_rows = herb_compound[herb_compound["CID_num"].isin(cids)].copy()
        herbs = sorted(herb_rows["Herb.Chinese.name"].dropna().astype(str).unique().tolist())
        related_formulae = sorted(
            formula_herb.loc[formula_herb["Herb.Chinese.name"].isin(herbs), "Formula.Chinese.name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        source_map = (
            herb_rows.groupby("CID_num")["Herb.Chinese.name"]
            .apply(lambda s: "、".join(sorted(set(s.dropna().astype(str)))))
            .to_dict()
        )
        chemicals["Source_Herbs"] = chemicals["CID_num"].map(source_map).fillna("")
        context = {
            "matched_compounds": chemicals["ChemicalName"].dropna().astype(str).unique().tolist(),
            "input_cids": cid_items,
            "input_smiles": smiles_items,
            "input_names": name_items,
            "herbs": herbs,
            "related_formulas": related_formulae,
        }
        return chemicals, context

    def _chemicals_from_herbs(self, herbs: list[str]) -> pd.DataFrame:
        data = self._load_data()
        herb_compound = data["herb_compound"]
        compounds = data["compound_class"]
        links = herb_compound[herb_compound["Herb.Chinese.name"].isin(herbs)].copy()
        if links.empty:
            return pd.DataFrame()
        cids = links["CID_num"].dropna().unique().tolist()
        chemicals = compounds[compounds["CID_num"].isin(cids)].copy()
        source_map = (
            links.groupby("CID_num")["Herb.Chinese.name"]
            .apply(lambda s: "、".join(sorted(set(s.dropna().astype(str)))))
            .to_dict()
        )
        chemicals["Source_Herbs"] = chemicals["CID_num"].map(source_map).fillna("")
        return chemicals

    def _normalize_chemical_frame(self, chemicals: pd.DataFrame) -> pd.DataFrame:
        df = chemicals.copy()
        if "Smiles" not in df.columns and "SMILES" in df.columns:
            df = df.rename(columns={"SMILES": "Smiles"})
        if "CID_num" not in df.columns and "CID" in df.columns:
            df["CID_num"] = pd.to_numeric(df["CID"], errors="coerce")
        for col in ["CID", "CID_num", "ChemicalName", "Smiles", "Class", "Superclass", "Pathway", "Is_glycoside", "Source_Herbs", "Source_Formulas"]:
            if col not in df.columns:
                df[col] = None
        df = df[pd.notna(df["Smiles"])].copy()
        df["Smiles"] = df["Smiles"].astype(str)
        df = df.drop_duplicates(subset=["Smiles"]).reset_index(drop=True)
        return df

    def _predict_intoblood(self, chemicals: pd.DataFrame) -> pd.DataFrame:
        reference = self._load_data()["intoblood_reference"]
        if reference.empty:
            raise PredictionError("入血参考库为空，无法进行入血筛选。", status_code=500)
        matched = match_intoblood_reference(chemicals, reference)
        matched["is_valid_smiles"] = matched["Smiles"].apply(
            lambda value: isinstance(value, str) and Chem.MolFromSmiles(value) is not None
        )
        return matched

    def _evaluate_toxicity(self, df: pd.DataFrame) -> pd.DataFrame:
        scored = df.copy()
        for col, default in [
            ("LogP", None),
            ("MW", None),
            ("QED", None),
            ("OB_Percent", None),
            ("Is_Absorbed", False),
            ("Pred_Cell_Toxicity", -1),
            ("Pred_Cell_prob", -1.0),
            ("Pred_Animal_Toxicity", -1),
            ("Pred_Animal_prob", -1.0),
            ("Pred_Clinical_Toxicity", -1),
            ("Pred_Clinical_prob", -1.0),
            ("Max_Tox_Prob", -1.0),
        ]:
            scored[col] = default

        for idx, row in scored.iterrows():
            if not bool(row.get("is_valid_smiles")):
                continue
            mol = Chem.MolFromSmiles(str(row["Smiles"]))
            if mol is None:
                continue
            logp = round(float(Descriptors.MolLogP(mol)), 2)
            mw = round(float(Descriptors.MolWt(mol)), 2)
            qed = round(float(QED.qed(mol)), 3)
            ob = _native(row.get("Bioavailability_Ma"))
            scored.at[idx, "LogP"] = logp
            scored.at[idx, "MW"] = mw
            scored.at[idx, "QED"] = qed
            scored.at[idx, "OB_Percent"] = ob
            scored.at[idx, "Is_Absorbed"] = bool(
                row.get("IntoBlood") == 1
                and logp <= 5.0
                and mw <= 500.0
                and qed >= 0.3
                and ob is not None
                and float(ob) > 0.3
            )

        absorbed_indices = scored[scored["Is_Absorbed"] == True].index
        if len(absorbed_indices) == 0:
            return scored

        smiles = scored.loc[absorbed_indices, "Smiles"].astype(str).tolist()
        pred_cell, prob_cell, pred_animal, prob_animal, pred_clinical, prob_clinical = self._predict_multimodel(smiles)
        scored.loc[absorbed_indices, "Pred_Cell_Toxicity"] = pred_cell
        scored.loc[absorbed_indices, "Pred_Cell_prob"] = np.round(prob_cell, 4)
        scored.loc[absorbed_indices, "Pred_Animal_Toxicity"] = pred_animal
        scored.loc[absorbed_indices, "Pred_Animal_prob"] = np.round(prob_animal, 4)
        scored.loc[absorbed_indices, "Pred_Clinical_Toxicity"] = pred_clinical
        scored.loc[absorbed_indices, "Pred_Clinical_prob"] = np.round(prob_clinical, 4)
        scored.loc[absorbed_indices, "Max_Tox_Prob"] = np.max([prob_cell, prob_animal, prob_clinical], axis=0)
        return scored

    def _load_models(self) -> tuple[Any, Any, Any]:
        if self._models is not None:
            return self._models
        with self._model_lock:
            if self._models is not None:
                return self._models
            if torch is None:
                raise PredictionError(
                    "缺少 torch/torch_geometric，无法加载模型包。请先安装 requirements.txt。",
                    status_code=500,
                )
            for path in (self.paths.model_cell, self.paths.model_animal, self.paths.model_clinical):
                if not path.exists():
                    raise PredictionError(f"缺少模型文件: {path}", status_code=500)
            try:
                self._models = load_model_bundles(
                    (
                        self.paths.model_cell,
                        self.paths.model_animal,
                        self.paths.model_clinical,
                    )
                )
            except Exception as exc:
                raise PredictionError(f"模型加载失败: {exc}", status_code=500) from exc
            return self._models

    def _predict_multimodel(self, smiles_list: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            return predict_model_bundles(smiles_list, self._load_models())
        except Exception as exc:
            raise PredictionError(f"多模型预测失败: {exc}", status_code=500) from exc

    def _build_prediction_response(
        self,
        query_type: str,
        items: list[str],
        scored: pd.DataFrame,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        threshold = 0.85
        valid = scored[scored["is_valid_smiles"] == True].copy()
        absorbed = valid[valid["Is_Absorbed"] == True].copy()
        high_toxic = absorbed[absorbed["Max_Tox_Prob"] >= threshold].sort_values("Max_Tox_Prob", ascending=False).copy()

        absorbed_count = int(len(absorbed))
        high_count = int(len(high_toxic))
        enrichment = high_count / absorbed_count if absorbed_count else 0.0
        risk = self._risk_from_enrichment(enrichment, absorbed_count)
        probabilities = {
            "cell": self._safe_max(absorbed["Pred_Cell_prob"]),
            "animal": self._safe_max(absorbed["Pred_Animal_prob"]),
            "clinical": self._safe_max(absorbed["Pred_Clinical_prob"]),
            "max": self._safe_max(absorbed["Max_Tox_Prob"]),
        }

        high_with_meta = high_toxic.copy()
        class_counts = _top_counts(high_with_meta.get("Class", pd.Series(dtype=str)))
        superclass_counts = _top_counts(high_with_meta.get("Superclass", pd.Series(dtype=str)))
        pathway_family_counts = _top_counts(high_with_meta.get("Pathway", pd.Series(dtype=str)))

        target_rows = self._target_rows_for_high_toxic(high_with_meta)
        pathway_rows = self._pathway_rows_for_targets(target_rows)
        go_rows = self._go_rows_for_targets(target_rows)
        disease_rows = self._disease_rows_for_targets(target_rows)

        display_cols = [
            "CID",
            "ChemicalName",
            "Smiles",
            "Source_Herbs",
            "Class",
            "Superclass",
            "Pathway",
            "Bioavailability_Ma",
            "Reference_Match",
            "IntoBlood",
            "IntoBlood_Reason",
            "LogP",
            "MW",
            "QED",
            "Is_Absorbed",
            "Pred_Cell_prob",
            "Pred_Animal_prob",
            "Pred_Clinical_prob",
            "Max_Tox_Prob",
        ]
        display_cols = [col for col in display_cols if col in scored.columns]
        trace_details = self._build_trace_details(query_type, context, scored)
        herb_candidates = self._summary_herb_candidates(query_type, items, trace_details, scored, context)
        food_medicine_homology = self._food_medicine_homology_summary(herb_candidates)
        consensus_reference = self._consensus_reference(absorbed, absorbed_count)
        toxicity_details = self._build_toxicity_details(
            high_toxic=high_toxic,
            display_cols=display_cols,
            target_rows=target_rows,
            pathway_rows=pathway_rows,
            go_rows=go_rows,
            disease_rows=disease_rows,
        )

        summary = {
            "total_compounds": int(len(scored)),
            "valid_smiles_count": int(len(valid)),
            "intoblood_count": int((valid["IntoBlood"] == 1).sum()),
            "absorbed_count": absorbed_count,
            "toxic_threshold": threshold,
            "high_toxic_count": high_count,
            "enrichment_ratio": round(float(enrichment), 6),
            "max_toxic_probability": probabilities["max"],
            "risk_level": risk["level"],
            "risk_label": risk["label"],
            "advice": risk["advice"],
            "food_medicine_homology": food_medicine_homology,
            "consensus_reference": consensus_reference,
        }

        return {
            "ok": True,
            "query_type": query_type,
            "query": items,
            "summary": summary,
            "probabilities": probabilities,
            "context": context,
            "warnings": self._warnings,
            "class_counts": class_counts,
            "superclass_counts": superclass_counts,
            "pathway_family_counts": pathway_family_counts,
            "target_counts": _top_counts(target_rows.get("Symbol", pd.Series(dtype=str))),
            "pathway_counts": _top_counts(pathway_rows.get("PathwayName", pd.Series(dtype=str))),
            "go_counts": _top_counts(go_rows.get("TERM", pd.Series(dtype=str))),
            "high_risk_compounds": _records(high_toxic[display_cols], 50),
            "compounds": _records(scored[display_cols].sort_values("Max_Tox_Prob", ascending=False), 100),
            "trace_details": trace_details,
            "toxicity_details": toxicity_details,
        }

    @staticmethod
    def _risk_from_enrichment(enrichment: float, absorbed_count: int) -> dict[str, str]:
        if absorbed_count == 0:
            return {
                "level": "low",
                "label": "低风险",
                "advice": "未检出满足入血与吸收阈值的有效成分，系统判定为低风险；建议结合实验数据复核。",
            }
        if enrichment < 0.02:
            return {
                "level": "low",
                "label": "低风险",
                "advice": "方剂或成分组整体肝毒性风险较低，可进入常规安全性复核流程。",
            }
        if enrichment < 0.05:
            return {
                "level": "moderate",
                "label": "中等风险",
                "advice": "存在肝毒性成分富集迹象，建议控制剂量并进行关键成分与肝功能指标复核。",
            }
        return {
            "level": "high",
            "label": "高风险",
            "advice": "系统性毒性富集率超过阈值，提示潜在药物性肝损伤风险，建议优先开展机制验证与配伍优化。",
        }

    @staticmethod
    def _normalize_herb_key(value: Any) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        text = str(value)
        text = text.replace("\u3000", " ").replace("\xa0", " ")
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
        text = re.sub(r"\s+", "", text)
        return text.strip()

    @staticmethod
    def _split_herb_names(value: Any) -> list[str]:
        if value is None:
            return []
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass
        text = str(value)
        pieces = re.split(r"[\n,，;；、/|]+", text)
        return [piece.strip() for piece in pieces if piece and piece.strip()]

    def _summary_herb_candidates(
        self,
        query_type: str,
        query: list[str],
        trace_details: dict[str, list[dict[str, Any]]],
        frame: pd.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        candidates: list[str] = []

        if query_type == "herb":
            candidates.extend(query)

        if context:
            for key in ("matched_herbs", "herbs"):
                values = context.get(key)
                if isinstance(values, list):
                    candidates.extend(str(value) for value in values)

        trace_sections: tuple[str, ...]
        if query_type == "formula":
            trace_sections = ("formula_herbs",)
        elif query_type == "compound":
            trace_sections = ("herb_compounds",)
        else:
            trace_sections = ("herb_compounds",)

        for section in trace_sections:
            for row in trace_details.get(section, []) or []:
                if not isinstance(row, dict):
                    continue
                for column in ("Herb.Chinese.name", "HerbName", "Herb"):
                    value = row.get(column)
                    if value:
                        candidates.extend(self._split_herb_names(value))

        if query_type == "compound" and frame is not None and not frame.empty:
            for column in ("Herb.Chinese.name", "Chinese_medicine_group", "Source_Herbs"):
                if column not in frame.columns:
                    continue
                for value in frame[column].dropna().unique().tolist():
                    candidates.extend(self._split_herb_names(value))

        seen: set[str] = set()
        cleaned: list[str] = []
        for candidate in candidates:
            key = self._normalize_herb_key(candidate)
            if not key or key in seen:
                continue
            seen.add(key)
            cleaned.append(key)
        return cleaned

    def _food_medicine_homology_summary(self, herb_candidates: list[str]) -> dict[str, Any]:
        source = "data/13_food_medicine_homology.csv"
        try:
            homology = self._load_data().get("food_medicine_homology", pd.DataFrame()).copy()
        except PredictionError as exc:
            return {
                "matched": False,
                "matched_herbs": [],
                "source": source,
                "note": f"药食同源名单不可用: {exc}",
                "matched_records": [],
            }

        if homology.empty or "HerbName" not in homology.columns:
            return {
                "matched": False,
                "matched_herbs": [],
                "source": source,
                "note": "药食同源名单为空或缺少 HerbName 字段。",
                "matched_records": [],
            }

        candidate_keys = {self._normalize_herb_key(value) for value in herb_candidates}
        candidate_keys.discard("")
        homology["_HerbKey"] = homology["HerbName"].map(self._normalize_herb_key)
        matched = homology[homology["_HerbKey"].isin(candidate_keys)].drop(columns=["_HerbKey"]).drop_duplicates()
        matched_herbs = sorted(
            matched["HerbName"].dropna().astype(str).map(self._normalize_herb_key).replace("", np.nan).dropna().unique().tolist()
        )
        if matched_herbs:
            note = "命中药食同源名单，仅作为研究参考标注，不改变肝毒性预测结果。"
        elif herb_candidates:
            note = "未命中药食同源名单，肝毒性预测结果照常供研究者参考。"
        else:
            note = "未识别到可用于药食同源匹配的药材名称。"
        return {
            "matched": bool(matched_herbs),
            "matched_herbs": matched_herbs,
            "source": source,
            "note": note,
            "matched_records": _records(matched, 50),
        }

    def _consensus_reference(self, df: pd.DataFrame, absorbed_count: int | None = None) -> dict[str, Any]:
        threshold = 0.75
        empty_result = {
            "max_probability": 0.0,
            "high_toxic_count": 0,
            "enrichment_ratio": 0.0,
            "risk_level": "low",
            "risk_label": "低风险",
            "advice": "未检出可用于模型一致性参考的有效吸收成分。",
            "toxic_threshold": threshold,
            "evaluated_count": 0,
            "animal_saturation_ratio": 0.0,
            "model_quality_flags": [],
        }
        if df.empty:
            return empty_result

        work = df.copy()
        if "Is_Absorbed" in work.columns:
            absorbed_mask = work["Is_Absorbed"].map(
                lambda value: value is True or str(value).strip().casefold() in {"true", "1", "yes"}
            )
            work = work[absorbed_mask].copy()
        if work.empty:
            return empty_result

        index = work.index
        weighted_sum = pd.Series(0.0, index=index)
        weight_sum = pd.Series(0.0, index=index)

        def add_weighted(column: str, weight: float, calibrated: bool = False) -> pd.Series:
            if column not in work.columns:
                return pd.Series(np.nan, index=index)
            values = pd.to_numeric(work[column], errors="coerce")
            values = values.where(values >= 0)
            if calibrated:
                values = (0.5 + 0.35 * (values - 0.5)).clip(lower=0.0, upper=1.0)
            valid = values.notna()
            weighted_sum.loc[valid] += values.loc[valid] * weight
            weight_sum.loc[valid] += weight
            return values

        add_weighted("Pred_Cell_prob", 0.35)
        animal_values = add_weighted("Pred_Animal_prob", 0.25, calibrated=True)
        add_weighted("Pred_Clinical_prob", 0.40)

        consensus = (weighted_sum / weight_sum).where(weight_sum > 0)
        consensus = consensus.dropna()
        if consensus.empty:
            return empty_result

        evaluated_count = int(len(consensus))
        denominator = int(absorbed_count if absorbed_count is not None else evaluated_count)
        if denominator <= 0:
            denominator = evaluated_count
        high_count = int((consensus >= threshold).sum())
        enrichment = high_count / denominator if denominator else 0.0
        risk = self._risk_from_enrichment(enrichment, denominator)

        animal_raw = pd.to_numeric(work.get("Pred_Animal_prob", pd.Series(dtype=float)), errors="coerce")
        animal_raw = animal_raw[animal_raw >= 0]
        saturation_ratio = float((animal_raw >= 0.99).sum() / len(animal_raw)) if len(animal_raw) else 0.0
        model_quality_flags: list[str] = []
        if saturation_ratio > 0.8:
            model_quality_flags.append("animal_probability_saturation")

        return {
            "max_probability": round(float(consensus.max()), 4),
            "high_toxic_count": high_count,
            "enrichment_ratio": round(float(enrichment), 6),
            "risk_level": risk["level"],
            "risk_label": risk["label"],
            "advice": risk["advice"],
            "toxic_threshold": threshold,
            "evaluated_count": evaluated_count,
            "animal_saturation_ratio": round(saturation_ratio, 6),
            "model_quality_flags": model_quality_flags,
        }

    @staticmethod
    def _safe_max(series: pd.Series) -> float:
        if series.empty:
            return 0.0
        numeric = pd.to_numeric(series, errors="coerce")
        numeric = numeric[numeric >= 0]
        if numeric.empty:
            return 0.0
        return round(float(numeric.max()), 4)

    def _targets_for_compounds(self, compounds: pd.DataFrame) -> list[str]:
        if compounds.empty or "ChemicalName" not in compounds.columns:
            return []
        data = self._load_data()
        names = compounds["ChemicalName"].dropna().astype(str).unique().tolist()
        targets = data["compound_target"]
        return sorted(targets[targets["ChemicalName"].isin(names)]["Symbol"].dropna().astype(str).unique().tolist())

    @staticmethod
    def _symbol_key(value: Any) -> str:
        return re.sub(r"\s+", "", str(value or "").strip()).upper()

    @staticmethod
    def _chem_key(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip()).casefold()

    @staticmethod
    def _entrez_text(value: Any) -> str:
        text = str(value or "").strip()
        if re.fullmatch(r"\d+(\.0+)?", text):
            return str(int(float(text)))
        return text

    def _ctd_meta_paths(self) -> dict[str, Path]:
        return {
            "target_disease": self.paths.target_disease,
            "chem_diseases": self.paths.chem_diseases,
            "chem_go": self.paths.chem_go,
            "chem_pathways": self.paths.chem_pathways,
            "compound_class": self.paths.compound_class,
            "compound_target": self.paths.compound_target,
            "target_pathway": self.paths.target_pathway,
            "target_go": self.paths.target_go,
        }

    @staticmethod
    def _meta_mtime_seconds(value: Any) -> int | None:
        if value is None:
            return None
        try:
            raw = int(float(str(value)))
        except (TypeError, ValueError):
            return None
        if abs(raw) > 10_000_000_000:
            return raw // 1_000_000_000
        return raw

    def _ctd_expected_meta(self) -> dict[str, str]:
        meta = {"schema_version": "4"}
        for key, path in self._ctd_meta_paths().items():
            stat = path.stat()
            meta[f"{key}.size"] = str(stat.st_size)
            meta[f"{key}.mtime_s"] = str(int(stat.st_mtime))
        return meta

    def _seed_ctd_cache_if_needed(self) -> None:
        if not is_frozen_app():
            return
        if self.paths.ctd_cache.exists():
            return
        if not self.paths.ctd_cache_seed.exists():
            return
        self.paths.ctd_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.paths.ctd_cache_seed, self.paths.ctd_cache)

    def _ctd_cache_is_current(self) -> bool:
        if not self.paths.ctd_cache.exists():
            return False
        try:
            with sqlite3.connect(self.paths.ctd_cache) as conn:
                rows = conn.execute("SELECT key, value FROM meta").fetchall()
            actual = {key: value for key, value in rows}
            if actual.get("schema_version") != "4":
                return False
            for key, path in self._ctd_meta_paths().items():
                stat = path.stat()
                if actual.get(f"{key}.size") != str(stat.st_size):
                    return False
                if not is_frozen_app():
                    actual_mtime = actual.get(f"{key}.mtime_s", actual.get(f"{key}.mtime_ns"))
                    if self._meta_mtime_seconds(actual_mtime) != int(stat.st_mtime):
                        return False
            return True
        except sqlite3.Error:
            return False

    def _ctd_local_keys(self, data: dict[str, pd.DataFrame]) -> tuple[set[str], set[str]]:
        symbol_keys: set[str] = set()
        for table_name in ("compound_target", "target_pathway", "target_go"):
            table = data.get(table_name, pd.DataFrame())
            if "Symbol" in table.columns:
                symbol_keys.update(
                    key for key in table["Symbol"].dropna().map(self._symbol_key).tolist() if key
                )

        chem_keys: set[str] = set()
        compounds = data.get("compound_class", pd.DataFrame())
        if "ChemicalName" in compounds.columns:
            chem_keys.update(
                key for key in compounds["ChemicalName"].dropna().map(self._chem_key).tolist() if key
            )
        return symbol_keys, chem_keys

    def _create_ctd_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE diseases (disease_id TEXT PRIMARY KEY);
            CREATE TABLE target_disease (
                symbol_key TEXT NOT NULL,
                Symbol TEXT,
                ENTREZID TEXT,
                DiseaseName TEXT,
                DiseaseID TEXT
            );
            CREATE TABLE chem_disease (
                chemical_key TEXT NOT NULL,
                ChemicalName TEXT,
                ChemicalID TEXT,
                DiseaseName TEXT,
                DiseaseID TEXT
            );
            CREATE TABLE chem_go (
                chemical_key TEXT NOT NULL,
                ChemicalName TEXT,
                ChemicalID TEXT,
                Ontology TEXT,
                GOTermID TEXT,
                GOTermName TEXT,
                PValue TEXT,
                CorrectedPValue TEXT
            );
            CREATE TABLE chem_pathway (
                chemical_key TEXT NOT NULL,
                ChemicalName TEXT,
                ChemicalID TEXT,
                PathwayID TEXT,
                PathwayName TEXT,
                PValue TEXT,
                CorrectedPValue TEXT
            );
            """
        )

    @staticmethod
    def _insert_disease_ids(conn: sqlite3.Connection, ids: pd.Series) -> None:
        values = [(str(value).strip(),) for value in ids.dropna().astype(str).unique() if str(value).strip()]
        if values:
            conn.executemany("INSERT OR IGNORE INTO diseases(disease_id) VALUES (?)", values)

    @staticmethod
    def _append_rows(conn: sqlite3.Connection, table: str, rows: pd.DataFrame) -> None:
        if not rows.empty:
            rows.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=500)

    def _populate_target_disease_cache(
        self,
        conn: sqlite3.Connection,
        local_symbols: set[str],
        chunksize: int = 200_000,
    ) -> None:
        usecols = ["Symbol", "ENTREZID", "DiseaseName", "DiseaseID"]
        for idx, chunk in enumerate(_read_csv_chunks_with_aliases(self.paths.target_disease, usecols, chunksize, dtype=str)):
            chunk = self._select_columns(chunk, usecols)
            self._insert_disease_ids(conn, chunk["DiseaseID"])
            chunk["symbol_key"] = chunk["Symbol"].map(self._symbol_key)
            filtered = chunk[chunk["symbol_key"].isin(local_symbols)].copy()
            if not filtered.empty:
                out = pd.DataFrame(
                    {
                        "symbol_key": filtered["symbol_key"],
                        "Symbol": filtered["Symbol"].astype(str).str.strip(),
                        "ENTREZID": filtered["ENTREZID"].map(self._entrez_text),
                        "DiseaseName": filtered["DiseaseName"].astype(str).str.strip(),
                        "DiseaseID": filtered["DiseaseID"].astype(str).str.strip(),
                    }
                )
                out = _drop_blank_relation_rows(out, ["DiseaseName", "DiseaseID"]).drop_duplicates()
                self._append_rows(conn, "target_disease", out)
            if idx % 10 == 0:
                conn.commit()

    def _populate_chem_disease_cache(
        self,
        conn: sqlite3.Connection,
        local_chems: set[str],
        chunksize: int = 200_000,
    ) -> None:
        usecols = ["ChemicalName", "ChemicalID", "DiseaseName", "DiseaseID"]
        for idx, chunk in enumerate(_read_csv_chunks_with_aliases(self.paths.chem_diseases, usecols, chunksize, dtype=str)):
            self._insert_disease_ids(conn, chunk["DiseaseID"])
            chunk["chemical_key"] = chunk["ChemicalName"].map(self._chem_key)
            filtered = chunk[chunk["chemical_key"].isin(local_chems)].copy()
            if not filtered.empty:
                out = filtered[["chemical_key", "ChemicalName", "ChemicalID", "DiseaseName", "DiseaseID"]].drop_duplicates()
                self._append_rows(conn, "chem_disease", out)
            if idx % 10 == 0:
                conn.commit()

    def _populate_chem_go_cache(
        self,
        conn: sqlite3.Connection,
        local_chems: set[str],
        chunksize: int = 200_000,
    ) -> None:
        usecols = ["ChemicalName", "ChemicalID", "Ontology", "GOTermName", "GOTermID", "PValue", "CorrectedPValue"]
        for idx, chunk in enumerate(_read_csv_chunks_with_aliases(self.paths.chem_go, usecols, chunksize, dtype=str)):
            chunk["chemical_key"] = chunk["ChemicalName"].map(self._chem_key)
            filtered = chunk[chunk["chemical_key"].isin(local_chems)].copy()
            if not filtered.empty:
                out = filtered[
                    ["chemical_key", "ChemicalName", "ChemicalID", "Ontology", "GOTermID", "GOTermName", "PValue", "CorrectedPValue"]
                ].drop_duplicates()
                self._append_rows(conn, "chem_go", out)
            if idx % 10 == 0:
                conn.commit()

    def _populate_chem_pathway_cache(
        self,
        conn: sqlite3.Connection,
        local_chems: set[str],
        chunksize: int = 200_000,
    ) -> None:
        usecols = ["ChemicalName", "ChemicalID", "PathwayName", "PathwayID", "PValue", "CorrectedPValue"]
        for idx, chunk in enumerate(_read_csv_chunks_with_aliases(self.paths.chem_pathways, usecols, chunksize, dtype=str)):
            chunk["chemical_key"] = chunk["ChemicalName"].map(self._chem_key)
            filtered = chunk[chunk["chemical_key"].isin(local_chems)].copy()
            if not filtered.empty:
                out = filtered[
                    ["chemical_key", "ChemicalName", "ChemicalID", "PathwayID", "PathwayName", "PValue", "CorrectedPValue"]
                ].drop_duplicates()
                self._append_rows(conn, "chem_pathway", out)
            if idx % 10 == 0:
                conn.commit()

    def _ensure_ctd_cache(self, data: dict[str, pd.DataFrame] | None = None) -> None:
        self._seed_ctd_cache_if_needed()
        if self._ctd_cache_is_current():
            return
        with self._ctd_cache_lock:
            self._seed_ctd_cache_if_needed()
            if self._ctd_cache_is_current():
                return
            data = data or self._load_data()
            local_symbols, local_chems = self._ctd_local_keys(data)
            tmp_path = self.paths.ctd_cache.with_name(f"{self.paths.ctd_cache.stem}.{os.getpid()}.tmp.sqlite")
            for path in (tmp_path,):
                if path.exists():
                    path.unlink()
            conn = sqlite3.connect(tmp_path)
            try:
                conn.execute("PRAGMA journal_mode=OFF")
                conn.execute("PRAGMA synchronous=OFF")
                conn.execute("PRAGMA temp_store=MEMORY")
                self._create_ctd_schema(conn)
                self._populate_target_disease_cache(conn, local_symbols)
                self._populate_chem_disease_cache(conn, local_chems)
                self._populate_chem_go_cache(conn, local_chems)
                self._populate_chem_pathway_cache(conn, local_chems)
                conn.executescript(
                    """
                    CREATE INDEX idx_target_disease_symbol ON target_disease(symbol_key);
                    CREATE INDEX idx_chem_disease_key ON chem_disease(chemical_key);
                    CREATE INDEX idx_chem_go_key ON chem_go(chemical_key);
                    CREATE INDEX idx_chem_pathway_key ON chem_pathway(chemical_key);
                    """
                )
                meta = self._ctd_expected_meta()
                conn.executemany(
                    "INSERT INTO meta(key, value) VALUES (?, ?)",
                    sorted(meta.items()),
                )
                conn.commit()
            finally:
                conn.close()
            self.paths.ctd_cache.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(30):
                try:
                    tmp_path.replace(self.paths.ctd_cache)
                    break
                except PermissionError:
                    if attempt == 29:
                        raise
                    time.sleep(1)

    def _ctd_select_by_keys(
        self,
        table: str,
        key_column: str,
        keys: list[str],
        columns: list[str],
        data: dict[str, pd.DataFrame] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        keys = sorted({key for key in keys if key})
        if not keys:
            return pd.DataFrame(columns=columns)
        self._ensure_ctd_cache(data)
        frames: list[pd.DataFrame] = []
        with sqlite3.connect(self.paths.ctd_cache) as conn:
            for start in range(0, len(keys), 400):
                if limit is not None:
                    current = sum(len(frame) for frame in frames)
                    if current >= limit:
                        break
                    remaining = limit - current
                else:
                    remaining = None
                subset = keys[start:start + 400]
                placeholders = ",".join("?" for _ in subset)
                sql = f"SELECT {', '.join(columns)} FROM {table} WHERE {key_column} IN ({placeholders})"
                if remaining is not None:
                    sql += f" LIMIT {int(remaining)}"
                frames.append(pd.read_sql_query(sql, conn, params=subset))
        if not frames:
            return pd.DataFrame(columns=columns)
        return pd.concat(frames, ignore_index=True).drop_duplicates()

    def _ctd_disease_count(self, data: dict[str, pd.DataFrame] | None = None) -> int:
        self._ensure_ctd_cache(data)
        with sqlite3.connect(self.paths.ctd_cache) as conn:
            row = conn.execute("SELECT COUNT(*) FROM diseases WHERE disease_id <> ''").fetchone()
        return int(row[0] if row else 0)

    def _ctd_cached_disease_count(self) -> int:
        self._seed_ctd_cache_if_needed()
        if self.paths.ctd_cache.exists():
            try:
                with sqlite3.connect(self.paths.ctd_cache) as conn:
                    row = conn.execute("SELECT COUNT(*) FROM diseases WHERE disease_id <> ''").fetchone()
                return int(row[0] if row else 0)
            except sqlite3.Error:
                pass
        return self._ctd_disease_count()

    def _disease_rows_for_symbols(
        self,
        symbols: Iterable[str],
        data: dict[str, pd.DataFrame] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        keys = [self._symbol_key(symbol) for symbol in symbols]
        return self._ctd_select_by_keys(
            "target_disease",
            "symbol_key",
            keys,
            ["Symbol", "ENTREZID", "DiseaseName", "DiseaseID"],
            data,
            limit,
        )

    def _chem_disease_rows_for_chemicals(
        self,
        chemicals: pd.DataFrame,
        data: dict[str, pd.DataFrame] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        if chemicals.empty or "ChemicalName" not in chemicals.columns:
            return pd.DataFrame()
        keys = chemicals["ChemicalName"].dropna().map(self._chem_key).tolist()
        return self._ctd_select_by_keys(
            "chem_disease",
            "chemical_key",
            keys,
            ["ChemicalName", "ChemicalID", "DiseaseName", "DiseaseID"],
            data,
            limit,
        )

    def _chem_go_rows_for_chemicals(
        self,
        chemicals: pd.DataFrame,
        data: dict[str, pd.DataFrame] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        if chemicals.empty or "ChemicalName" not in chemicals.columns:
            return pd.DataFrame()
        keys = chemicals["ChemicalName"].dropna().map(self._chem_key).tolist()
        return self._ctd_select_by_keys(
            "chem_go",
            "chemical_key",
            keys,
            ["ChemicalName", "ChemicalID", "Ontology", "GOTermID", "GOTermName", "PValue", "CorrectedPValue"],
            data,
            limit,
        )

    def _chem_pathway_rows_for_chemicals(
        self,
        chemicals: pd.DataFrame,
        data: dict[str, pd.DataFrame] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        if chemicals.empty or "ChemicalName" not in chemicals.columns:
            return pd.DataFrame()
        keys = chemicals["ChemicalName"].dropna().map(self._chem_key).tolist()
        return self._ctd_select_by_keys(
            "chem_pathway",
            "chemical_key",
            keys,
            ["ChemicalName", "ChemicalID", "PathwayID", "PathwayName", "PValue", "CorrectedPValue"],
            data,
            limit,
        )

    def _target_rows_for_high_toxic(self, high_toxic: pd.DataFrame) -> pd.DataFrame:
        if high_toxic.empty:
            return pd.DataFrame()
        data = self._load_data()
        high_toxic = self._fill_compound_names_from_cid(high_toxic)
        names = high_toxic.get("ChemicalName", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
        if not names:
            return pd.DataFrame()
        return data["compound_target"][data["compound_target"]["ChemicalName"].isin(names)].drop_duplicates().copy()

    def _pathway_rows_for_targets(self, target_rows: pd.DataFrame) -> pd.DataFrame:
        if target_rows.empty or "Symbol" not in target_rows.columns:
            return pd.DataFrame()
        data = self._load_data()
        symbols = target_rows["Symbol"].dropna().astype(str).unique().tolist()
        return data["target_pathway"][data["target_pathway"]["Symbol"].isin(symbols)].drop_duplicates().copy()

    def _go_rows_for_targets(self, target_rows: pd.DataFrame) -> pd.DataFrame:
        if target_rows.empty or "Symbol" not in target_rows.columns:
            return pd.DataFrame()
        data = self._load_data()
        symbols = target_rows["Symbol"].dropna().astype(str).unique().tolist()
        rows = data["target_go"][data["target_go"]["Symbol"].isin(symbols)].merge(data["go_term"], on="GOID", how="left")
        return rows.drop_duplicates().copy()

    def _disease_rows_for_targets(self, target_rows: pd.DataFrame) -> pd.DataFrame:
        if target_rows.empty or "Symbol" not in target_rows.columns:
            return pd.DataFrame()
        symbols = target_rows["Symbol"].dropna().astype(str).unique().tolist()
        return self._disease_rows_for_symbols(symbols, limit=1000)

    def _with_entrez(self, rows: pd.DataFrame) -> pd.DataFrame:
        if rows.empty or ("Symbol" not in rows.columns and "ENTREZID" not in rows.columns):
            return rows.copy()
        return self._fill_target_identifiers(rows)

    @staticmethod
    def _select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = _coalesce_relation_columns(df)
        existing = [col for col in columns if col in df.columns]
        if not existing:
            return pd.DataFrame(columns=columns)
        return df[existing].drop_duplicates().copy()

    def _build_trace_details(
        self,
        query_type: str,
        context: dict[str, Any],
        scored: pd.DataFrame,
    ) -> dict[str, list[dict[str, Any]]]:
        data = self._load_data()
        formula_herb = data["formula_herb"]
        herb_compound = data["herb_compound"]

        matched_formulas = context.get("matched_formulas") or context.get("related_formulas") or []
        matched_herbs = context.get("matched_herbs") or context.get("herbs") or []

        formula_rows = pd.DataFrame()
        if matched_formulas:
            formula_rows = formula_herb[formula_herb["Formula.Chinese.name"].isin(matched_formulas)].copy()
        if formula_rows.empty and matched_herbs:
            formula_rows = formula_herb[formula_herb["Herb.Chinese.name"].isin(matched_herbs)].copy()

        cid_series = pd.to_numeric(scored.get("CID_num", pd.Series(dtype=float)), errors="coerce")
        cids = cid_series.dropna().unique().tolist()
        herb_rows = herb_compound[herb_compound["CID_num"].isin(cids)].copy() if cids else pd.DataFrame()

        # 方剂/中药预测时，只展示本次检索命中的中药，不展示共享同一成分的全库中药
        if query_type in ("formula", "herb") and matched_herbs and not herb_rows.empty:
            herb_rows = herb_rows[
                herb_rows["Herb.Chinese.name"].isin(matched_herbs)
            ].copy()

        target_rows = self._target_rows_for_high_toxic(scored)
        target_rows_with_entrez = self._with_entrez(target_rows)
        pathway_rows = self._pathway_rows_for_targets(target_rows)
        go_rows = self._go_rows_for_targets(target_rows)
        disease_rows = self._with_entrez(self._disease_rows_for_targets(target_rows))
        chem_disease_rows = self._chem_disease_rows_for_chemicals(scored, data, limit=1000)
        chem_go_rows = self._chem_go_rows_for_chemicals(scored, data, limit=1000)
        chem_pathway_rows = self._chem_pathway_rows_for_chemicals(scored, data, limit=1000)

        compound_columns = [
            "CID",
            "ChemicalName",
            "Source_Herbs",
            "Source_Formulas",
            "Class",
            "Superclass",
            "Pathway",
            "Is_glycoside",
        ]

        return {
            "formula_herbs": _records(
                self._select_columns(formula_rows, ["Formula.Chinese.name", "Herb.Chinese.name"]),
                1000,
            ),
            "herb_compounds": _records(
                self._select_columns(herb_rows, ["Herb.Chinese.name", "Herb.Pinyin.name", "CID", "ChemicalName"]),
                1000,
            ),
            "compound_classes": _balanced_records(self._select_columns(scored, compound_columns), ["ChemicalName", "CID"], 1000),
            "compound_targets": _balanced_records(
                self._select_columns(target_rows_with_entrez, ["ChemicalName", "Symbol", "ENTREZID"]),
                ["ChemicalName"],
                1000,
            ),
            "chem_diseases": _balanced_records(
                self._select_columns(chem_disease_rows, ["ChemicalName", "ChemicalID", "DiseaseName", "DiseaseID"]),
                ["ChemicalName"],
                1000,
            ),
            "chem_go": _balanced_records(
                self._select_columns(chem_go_rows, ["ChemicalName", "ChemicalID", "Ontology", "GOTermID", "GOTermName", "PValue", "CorrectedPValue"]),
                ["ChemicalName"],
                1000,
            ),
            "chem_pathways": _balanced_records(
                self._select_columns(chem_pathway_rows, ["ChemicalName", "ChemicalID", "PathwayID", "PathwayName", "PValue", "CorrectedPValue"]),
                ["ChemicalName"],
                1000,
            ),
            "target_go": _balanced_records(self._select_columns(go_rows, ["Symbol", "GOID", "Ontology", "TERM"]), ["Symbol"], 1000),
            "target_pathways": _balanced_records(
                self._select_columns(pathway_rows, ["ENTREZID", "Symbol", "Pathwayid", "PathwayName"]),
                ["Symbol"],
                1000,
            ),
            "target_diseases": _balanced_records(
                self._select_columns(disease_rows, ["ENTREZID", "Symbol", "DiseaseName", "DiseaseID"]),
                ["Symbol"],
                1000,
            ),
        }

    def _build_toxicity_details(
        self,
        high_toxic: pd.DataFrame,
        display_cols: list[str],
        target_rows: pd.DataFrame,
        pathway_rows: pd.DataFrame,
        go_rows: pd.DataFrame,
        disease_rows: pd.DataFrame,
    ) -> dict[str, list[dict[str, Any]]]:
        high_sorted = high_toxic.sort_values("Max_Tox_Prob", ascending=False).copy() if not high_toxic.empty else high_toxic.copy()
        toxic_meta = self._select_columns(high_sorted, ["ChemicalName", "Max_Tox_Prob"])

        target_links = self._with_entrez(target_rows)
        if not toxic_meta.empty and not target_links.empty:
            target_links = target_links.merge(toxic_meta, on="ChemicalName", how="left")

        target_bridge = self._select_columns(target_rows, ["ChemicalName", "Symbol"])
        go_links = target_bridge.merge(go_rows, on="Symbol", how="inner") if not target_bridge.empty and not go_rows.empty else pd.DataFrame()
        pathway_links = target_bridge.merge(pathway_rows, on="Symbol", how="inner") if not target_bridge.empty and not pathway_rows.empty else pd.DataFrame()
        disease_links = (
            target_bridge.merge(disease_rows, on="Symbol", how="inner")
            if not target_bridge.empty and not disease_rows.empty
            else pd.DataFrame()
        )

        return {
            "toxic_compounds": _records(self._select_columns(high_sorted, display_cols), 200),
            "toxic_targets": _records(
                self._select_columns(target_links, ["ChemicalName", "Symbol", "ENTREZID", "Max_Tox_Prob"]),
                500,
            ),
            "toxic_target_go": _records(
                self._select_columns(go_links, SECTION_CORE_COLUMNS["toxic_go"]),
                500,
            ),
            "toxic_target_pathways": _records(
                self._select_columns(pathway_links, SECTION_CORE_COLUMNS["toxic_pathways"]),
                500,
            ),
            "toxic_target_diseases": _records(
                self._select_columns(disease_links, ["ChemicalName", "ENTREZID", "Symbol", "DiseaseName", "DiseaseID"]),
                500,
            ),
        }

    @staticmethod
    def _join_counts(items: list[dict[str, Any]]) -> str:
        if not items:
            return "未检出"
        return "、".join(f"{item['name']}({item['count']})" for item in items)

    @staticmethod
    def _relaxed_formula_terms(items: list[str]) -> list[str]:
        dosage_suffixes = ("丸", "散", "片", "颗粒", "胶囊", "合剂", "口服液", "糖浆", "膏", "酊")
        terms: list[str] = []
        for item in items:
            clean = item.strip()
            for suffix in dosage_suffixes:
                if clean.endswith(suffix) and len(clean) > len(suffix) + 1:
                    terms.append(clean[: -len(suffix)])
            if len(clean) >= 3:
                terms.append(clean[:-1])
        return sorted(set(term for term in terms if len(term) >= 2))
