# %%
import os
import sys
from pathlib import Path

import pandas as pd

__all__ = [
    "data_01",
    "data_02",
    "data_03",
    "data_04",
    "data_05",
    "data_06",
    "data_07",
    "data_08",
    "data_09",
    "data_10",
    "data_11",
    "copy_column_aliases",
]


def resource_root() -> Path:
    override = os.getenv("TOXHERB_RESOURCE_ROOT")
    if override:
        return Path(override).resolve()
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root).resolve()
    return Path(__file__).resolve().parents[1]


DATA_DIR = resource_root() / "data"


COLUMN_ALIASES = {
    "TERM": ("GOTerm", "GOTermName", "term"),
    "GOTermName": ("GOTerm", "TERM", "term"),
    "GOTermID": ("GOID", "goid"),
    "HerbName": ("Herb.Chinese.name", "Herb", "ChineseName"),
    "GeneSymbol": ("Symbol", "genesymbol", "Target"),
    "GeneID": ("ENTREZID", "EntrezID", "entrezid"),
    "Pathwayid": ("PathwayID", "pathwayid"),
}


def copy_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for canonical, candidates in COLUMN_ALIASES.items():
        if canonical in out.columns:
            continue
        source = next((candidate for candidate in candidates if candidate in out.columns), None)
        if source is not None:
            out[canonical] = out[source]
    return out


def read_data_csv(name: str, **kwargs) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    try:
        return copy_column_aliases(pd.read_csv(path, **kwargs))
    except UnicodeDecodeError:
        return copy_column_aliases(pd.read_csv(path, encoding="gbk", **kwargs))


data_01 = read_data_csv("01_formula_herb.csv")
data_02 = read_data_csv("02_herb_compound.csv")
data_03 = read_data_csv("03_compound_class.csv").drop("ChemicalName", axis=1)
if "SMILES" not in data_03.columns and "Smiles" in data_03.columns:
    data_03 = data_03.rename(columns={"Smiles": "SMILES"})
data_04 = read_data_csv("04_compound_target.csv")
data_05 = read_data_csv("05_target_pathway.csv")
data_06 = read_data_csv("06_target_go.csv")
data_07 = read_data_csv("07_go_term.csv")
data_08 = read_data_csv("10_chem_go.csv")
data_09 = read_data_csv("09_chem_diseases.csv")
data_10 = read_data_csv("11_chem_pathways.csv")
data_11 = read_data_csv("08_target_disease.csv")
data_11 = data_11.drop(columns=[col for col in ("Symbol", "ENTREZID") if col in data_11.columns])


if __name__ == "__main__":
    for name in __all__:
        frame = globals()[name]
        print(f"{name}: {len(frame)} rows, {frame.shape[1]} columns")
