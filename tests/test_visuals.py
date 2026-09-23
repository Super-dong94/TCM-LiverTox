from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backend import predictor as predictor_module
from backend.predictor import HepatotoxicityPredictor


REQUIRED_CHART_FIELDS = {"id", "title", "type", "data", "encoding", "source_section"}


def write_minimal_data_files(root: Path) -> None:
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "01_formula_herb.csv").write_text(
        "Formula.Chinese.name,Herb.Chinese.name\n阿胶地黄汤,九里香\n",
        encoding="utf-8",
    )
    (data / "02_herb_compound.csv").write_text(
        "Herb.Pinyin.name,Herb.Chinese.name,ChemicalName,CID\njiulixiang,九里香,Emodin,15945058\n",
        encoding="utf-8",
    )
    (data / "03_compound_class.csv").write_text(
        "CID,ChemicalName,SMILES,Class,Superclass,Pathway,Is_glycoside\n"
        "15945058,Emodin,C,Anthraquinones,Phenolics,Shikimates,0\n",
        encoding="utf-8",
    )
    (data / "04_compound_target.csv").write_text(
        "ChemicalName,Symbol\nEmodin,CYP3A4\n",
        encoding="utf-8",
    )
    (data / "05_target_pathway.csv").write_text(
        "ENTREZID,Symbol,Pathwayid,PathwayName\n1576,CYP3A4,hsa00982,Drug metabolism\n",
        encoding="utf-8",
    )
    (data / "06_target_go.csv").write_text(
        "Symbol,GOID,Ontology\nCYP3A4,GO:0006805,BP\n",
        encoding="utf-8",
    )
    (data / "07_go_term.csv").write_text(
        "GOID,GOTerm\nGO:0006805,xenobiotic metabolic process\n",
        encoding="utf-8",
    )
    (data / "08_target_disease.csv").write_text(
        "Symbol,ENTREZID,DiseaseName,DiseaseID\nCYP3A4,1576,Drug-induced liver injury,MESH:D056486\n",
        encoding="utf-8",
    )
    (data / "09_chem_diseases.csv").write_text(
        "ChemicalName,ChemicalID,CasRN,DiseaseName,DiseaseID\nEmodin,C000000,518-82-1,Liver injury,MESH:D008107\n",
        encoding="utf-8",
    )
    (data / "10_chem_go.csv").write_text(
        "ChemicalName,ChemicalID,CasRN,Ontology,GOTerm,GOID,HighestGOLevel,PValue,CorrectedPValue\n"
        "Emodin,C000000,518-82-1,BP,xenobiotic metabolic process,GO:0006805,5,0.01,0.02\n",
        encoding="utf-8",
    )
    (data / "11_chem_pathways.csv").write_text(
        "ChemicalName,ChemicalID,CasRN,PathwayName,PathwayID,PValue,CorrectedPValue\n"
        "Emodin,C000000,518-82-1,Drug metabolism,REACT:R-HSA-211981,0.01,0.02\n",
        encoding="utf-8",
    )
    (data / "12_intoblood_ref.csv").write_text(
        "CID,ChemicalName,Smiles,Bioavailability_Ma,Reference_Match,IntoBlood,IntoBlood_Reason\n"
        "15945058,Emodin,C,0.8,1,1,reference_match\n",
        encoding="utf-8",
    )
    (data / "13_food_medicine_homology.csv").write_text(
        "Herb.Chinese.name,SourceName,Announcement,LimitNote,MedicinalPart,Dose,Rank,Class1,Class2\n"
        "九里香,Test,Test,,root,,1,食品,药食同源\n",
        encoding="utf-8",
    )


def assert_chart_specs(specs: list[dict[str, object]]) -> None:
    assert specs
    for spec in specs:
        assert REQUIRED_CHART_FIELDS.issubset(spec)
        assert isinstance(spec["data"], list)
        assert isinstance(spec["encoding"], dict)
        if spec["data"] and spec["type"] != "graph":
            row = spec["data"][0]
            assert isinstance(row, dict)
            for field in spec["encoding"].values():
                assert field in row or field in {"nodes", "edges"}


def test_load_data_adds_unified_column_aliases(tmp_path) -> None:
    write_minimal_data_files(tmp_path)
    predictor = HepatotoxicityPredictor(root=tmp_path)

    data = predictor._load_data(refresh=True)

    assert data["go_term"].loc[0, "TERM"] == "xenobiotic metabolic process"
    assert data["intoblood_reference"].loc[0, "Smiles"] == "C"
    assert data["food_medicine_homology"].loc[0, "HerbName"] == "九里香"
    assert data["compound_class"].loc[0, "Smiles"] == "C"


def test_ctd_cache_rebuild_supports_unified_column_names(tmp_path) -> None:
    write_minimal_data_files(tmp_path)
    predictor = HepatotoxicityPredictor(root=tmp_path)
    data = predictor._load_data(refresh=True)

    predictor._ensure_ctd_cache(data)

    with sqlite3.connect(predictor.paths.ctd_cache) as conn:
        target_row = conn.execute(
            "SELECT Symbol, ENTREZID, DiseaseName FROM target_disease"
        ).fetchone()
        chem_go_row = conn.execute(
            "SELECT ChemicalName, GOTermID, GOTermName FROM chem_go"
        ).fetchone()
        schema_version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
    assert target_row == ("CYP3A4", "1576", "Drug-induced liver injury")
    assert chem_go_row == ("Emodin", "GO:0006805", "xenobiotic metabolic process")
    assert schema_version == "4"


def test_prediction_cache_key_includes_response_schema_version(tmp_path, monkeypatch) -> None:
    write_minimal_data_files(tmp_path)
    predictor = HepatotoxicityPredictor(root=tmp_path)

    key = predictor.prediction_cache_key("herb", ["何首乌"])
    monkeypatch.setattr(predictor_module, "PREDICTION_RESPONSE_SCHEMA_VERSION", "changed-schema")

    assert predictor.prediction_cache_key("herb", ["何首乌"]) != key


def test_external_prediction_frames_rebuild_compound_targets_from_cid(tmp_path) -> None:
    write_minimal_data_files(tmp_path)
    predictor = HepatotoxicityPredictor(root=tmp_path)
    livertox = pd.DataFrame([{"CID": "15945058", "Max_Tox_Prob": 0.91}])
    high_toxic = pd.DataFrame([{"CID": "15945058", "Max_Tox_Prob": 0.91}])
    empty_targets = pd.DataFrame(columns=["ChemicalName", "Symbol"])

    _, high_out, target_out, pathway_out, go_out, _ = predictor._augment_external_prediction_frames(
        livertox,
        high_toxic,
        empty_targets,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )

    assert high_out.loc[0, "ChemicalName"] == "Emodin"
    assert target_out[["ChemicalName", "Symbol", "ENTREZID"]].iloc[0].to_dict() == {
        "ChemicalName": "Emodin",
        "Symbol": "CYP3A4",
        "ENTREZID": "1576",
    }
    assert pathway_out.iloc[0]["ENTREZID"] == "1576"
    assert go_out.iloc[0]["GOID"] == "GO:0006805"


def test_compound_mechanism_section_page_rebuilds_empty_target_csv(tmp_path) -> None:
    write_minimal_data_files(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "06_high_toxic_chemicals_master.csv").write_text(
        "CID,Max_Tox_Prob\n15945058,0.91\n",
        encoding="utf-8",
    )
    (output_dir / "07_high_toxic_chemical_targets.csv").write_text(
        "ChemicalName,Symbol\n",
        encoding="utf-8",
    )
    (output_dir / "08_high_toxic_target_pathways.csv").write_text(
        "Symbol,ENTREZID,Pathwayid,PathwayName\n",
        encoding="utf-8",
    )
    (output_dir / "09_high_toxic_target_GO_terms.csv").write_text(
        "Symbol,GOID,Ontology,TERM\n",
        encoding="utf-8",
    )
    (output_dir / "10_high_toxic_target_disease_results.csv").write_text(
        "Symbol,GeneID,DiseaseName,DiseaseID\n",
        encoding="utf-8",
    )
    predictor = HepatotoxicityPredictor(root=tmp_path)

    targets = predictor.get_output_section_page(
        query_type="compound",
        output_dir=output_dir,
        section_id="toxic_targets",
        page=1,
        page_size=10,
    )
    pathways = predictor.get_output_section_page(
        query_type="compound",
        output_dir=output_dir,
        section_id="toxic_pathways",
        page=1,
        page_size=10,
    )
    diseases = predictor.get_output_section_page(
        query_type="compound",
        output_dir=output_dir,
        section_id="toxic_diseases",
        page=1,
        page_size=10,
    )

    assert targets["section"]["row_count"] == 1
    assert targets["records"][0]["ChemicalName"] == "Emodin"
    assert targets["records"][0]["Symbol"] == "CYP3A4"
    assert targets["records"][0]["ENTREZID"] == "1576"
    assert pathways["section"]["row_count"] == 1
    assert pathways["records"][0]["ENTREZID"] == "1576"
    assert diseases["section"]["row_count"] == 1
    assert diseases["records"][0]["ChemicalName"] == "Emodin"
    assert diseases["records"][0]["ENTREZID"] == "1576"
    assert diseases["records"][0]["DiseaseName"] == "Drug-induced liver injury"


def test_select_columns_coalesces_relation_aliases() -> None:
    source = pd.DataFrame(
        [
            {
                "ChemicalName": "Emodin",
                "GeneSymbol": "CYP3A4",
                "GeneID": "1576",
                "GOTermID_x": "GO:0006805",
                "GOTermName_y": "xenobiotic metabolic process",
                "PathwayID": "hsa00982",
                "DiseaseName": "Drug-induced liver injury",
                "DiseaseID": "MESH:D056486",
            }
        ]
    )

    selected = HepatotoxicityPredictor._select_columns(
        source,
        ["ChemicalName", "Symbol", "ENTREZID", "GOID", "TERM", "Pathwayid", "DiseaseName", "DiseaseID"],
    )

    assert selected.iloc[0].to_dict() == {
        "ChemicalName": "Emodin",
        "Symbol": "CYP3A4",
        "ENTREZID": "1576",
        "GOID": "GO:0006805",
        "TERM": "xenobiotic metabolic process",
        "Pathwayid": "hsa00982",
        "DiseaseName": "Drug-induced liver injury",
        "DiseaseID": "MESH:D056486",
    }


def test_select_columns_coalesces_disease_suffix_aliases() -> None:
    source = pd.DataFrame(
        [
            {
                "ChemicalName": "Emodin",
                "Symbol": "CYP3A4",
                "ENTREZID": "未提供",
                "ENTREZID_disease": "1576",
                "GeneID_disease": "1576",
                "DiseaseName": "Drug-induced liver injury",
                "DiseaseID": "MESH:D056486",
            }
        ]
    )

    selected = HepatotoxicityPredictor._select_columns(
        source,
        ["ChemicalName", "ENTREZID", "Symbol", "DiseaseName", "DiseaseID"],
    )

    assert selected.iloc[0].to_dict() == {
        "ChemicalName": "Emodin",
        "ENTREZID": "1576",
        "Symbol": "CYP3A4",
        "DiseaseName": "Drug-induced liver injury",
        "DiseaseID": "MESH:D056486",
    }


def test_build_toxicity_details_uses_target_disease_links() -> None:
    predictor = HepatotoxicityPredictor.__new__(HepatotoxicityPredictor)
    high_toxic = pd.DataFrame([{"ChemicalName": "Emodin", "Max_Tox_Prob": 0.91}])
    target_rows = pd.DataFrame([{"ChemicalName": "Emodin", "Symbol": "CYP3A4", "ENTREZID": "1576"}])
    pathway_rows = pd.DataFrame([{"Symbol": "CYP3A4", "ENTREZID": "1576", "PathwayID": "hsa00982", "PathwayName": "Drug metabolism"}])
    go_rows = pd.DataFrame([{"Symbol": "CYP3A4", "GOID": "GO:0006805", "Ontology": "BP", "GOTermName": "xenobiotic metabolic process"}])
    disease_rows = pd.DataFrame(
        [{"Symbol": "CYP3A4", "ENTREZID": "1576", "DiseaseName": "Drug-induced liver injury", "DiseaseID": "MESH:D056486"}]
    )

    details = predictor._build_toxicity_details(
        high_toxic,
        ["ChemicalName", "Max_Tox_Prob"],
        target_rows,
        pathway_rows,
        go_rows,
        disease_rows,
    )

    assert set(details["toxic_target_go"][0]) == {"Symbol", "GOID", "Ontology", "TERM"}
    assert set(details["toxic_target_pathways"][0]) == {"Symbol", "ENTREZID", "Pathwayid", "PathwayName"}
    disease_record = details["toxic_target_diseases"][0]
    assert disease_record["ChemicalName"] == "Emodin"
    assert disease_record["Symbol"] == "CYP3A4"
    assert disease_record["ENTREZID"] == "1576"
    assert disease_record["DiseaseName"] == "Drug-induced liver injury"
    assert disease_record["DiseaseID"] == "MESH:D056486"
    assert "ChemicalID" not in disease_record


def test_toxic_go_section_preview_projects_core_columns(tmp_path) -> None:
    output_dir = tmp_path
    csv_path = output_dir / "09_high_toxic_target_GO_terms.csv"
    csv_path.write_text(
        "Symbol,GOID,Ontology,GOTermID_x,GeneSymbol,GOTerm,TERM,GOTermName\n"
        "CYP3A4,GO:0006805,BP,GO:0006805,CYP3A4,xenobiotic metabolic process,xenobiotic metabolic process,xenobiotic metabolic process\n",
        encoding="utf-8",
    )
    predictor = HepatotoxicityPredictor.__new__(HepatotoxicityPredictor)

    section = predictor._section_from_csv("毒性靶标-GO术语关系", csv_path, "toxic_go")

    assert section["headers"] == ["Symbol", "GOID", "Ontology", "TERM"]
    assert section["rows"][0] == ["CYP3A4", "GO:0006805", "BP", "xenobiotic metabolic process"]


def test_prediction_visual_specs_schema() -> None:
    predictor = HepatotoxicityPredictor.__new__(HepatotoxicityPredictor)
    high = pd.DataFrame(
        [
            {
                "ChemicalName": "Emodin",
                "Max_Tox_Prob": 0.91,
                "compound_priority_score": 0.88,
                "Class": "Anthraquinones",
                "Source_Herbs": "何首乌",
            }
        ]
    )
    targets = pd.DataFrame([{"ChemicalName": "Emodin", "Symbol": "CYP3A4", "target_priority_score": 0.9}])
    pathways = pd.DataFrame([{"Symbol": "CYP3A4", "PathwayName": "Drug metabolism", "pathway_priority_score": 0.8, "hit_count": 1}])
    go = pd.DataFrame([{"TERM": "oxidative stress response", "go_priority_score": 0.7, "go_category": "BP"}])
    diseases = pd.DataFrame([{"DiseaseName": "Drug-induced liver injury", "disease_priority_score": 0.9}])
    specs = predictor.build_prediction_visuals_from_frames(
        {"total_compounds": 1, "valid_smiles_count": 1, "intoblood_count": 1, "absorbed_count": 1, "high_toxic_count": 1},
        {"cell": 0.7, "animal": 0.8, "clinical": 0.6},
        high,
        targets,
        pathways,
        go,
        diseases,
    )
    assert_chart_specs(specs)


def test_chart_spec_helper_schema() -> None:
    spec = HepatotoxicityPredictor._chart_spec(
        "database_scale_bar",
        "数据库规模",
        "bar",
        [{"entity": "成分", "count": 10}],
        {"x": "entity", "y": "count"},
        source_section="database",
    )
    assert_chart_specs([spec])


def test_database_visuals_overview_includes_chemical_category_stats() -> None:
    predictor = HepatotoxicityPredictor.__new__(HepatotoxicityPredictor)
    predictor.paths = SimpleNamespace(root=Path("."))  # type: ignore[attr-defined]
    stats = {
        "formulas": 10,
        "herbs": 20,
        "compounds": 30,
        "classes": 4,
        "superclasses": 2,
        "chemical_pathways": 1,
        "targets": 40,
        "signal_pathways": 5,
        "go_terms": 6,
        "diseases": 7,
    }
    predictor.database_stats = lambda: {"ok": True, "stats": stats}  # type: ignore[method-assign]
    predictor._load_data = lambda: {  # type: ignore[method-assign]
        "compound_class": pd.DataFrame({"Class": ["A", "B", "A"]}),
        "compound_target": pd.DataFrame({"Symbol": ["CYP3A4", "CYP3A4", "RELA"]}),
        "target_pathway": pd.DataFrame({"PathwayName": ["Drug metabolism", "Apoptosis"]}),
    }
    predictor._load_manifest = lambda path: {"schema_version": "test", "updated_at": "now", "hash": "abcdef"}  # type: ignore[method-assign]

    result = predictor.build_database_visuals()
    specs = result["chart_specs"]
    overview = specs[0]
    spec_ids = {spec["id"] for spec in specs}

    assert overview["id"] == "database_overview_bar"
    assert overview["title"] == "数据库信息总览"
    entities = {row["entity"]: row["count"] for row in overview["data"]}
    assert entities["化学类别"] == 4
    assert entities["化学超类"] == 2
    assert entities["化学通路"] == 1
    assert len(overview["data"]) == 10
    assert {"data_source_total_bar", "entity_source_grouped_bar", "source_cleaning_result_bar"}.issubset(spec_ids)
    assert {"target_degree_top", "pathway_top", "data_version_card"}.isdisjoint(spec_ids)
    assert_chart_specs(specs)


def test_database_source_visuals_use_word_source_table() -> None:
    predictor = HepatotoxicityPredictor.__new__(HepatotoxicityPredictor)
    predictor.paths = SimpleNamespace(root=Path("."))  # type: ignore[attr-defined]
    predictor.database_stats = lambda: {"ok": True, "stats": {}}  # type: ignore[method-assign]
    predictor._load_data = lambda: {"compound_class": pd.DataFrame({"Class": ["Flavonoids"]})}  # type: ignore[method-assign]
    predictor._load_manifest = lambda path: {"schema_version": "test"}  # type: ignore[method-assign]

    result = predictor.build_database_visuals()
    specs = {spec["id"]: spec for spec in result["chart_specs"]}
    source_data = specs["data_source_total_bar"]["data"]
    entity_source_data = specs["entity_source_grouped_bar"]["data"]
    cleaning_data = specs["source_cleaning_result_bar"]["data"]

    sources = {row["source"] for row in source_data}
    assert {"BATMAN-TCM", "DCABM-TCM", "TCMSP", "CTD", "NPClassifier", "中国药典2025版", "中医世家"}.issubset(sources)
    assert {"x": "source", "y": "count", "series": "entity"} == specs["data_source_total_bar"]["encoding"]
    assert {"x": "entity", "y": "count", "series": "source"} == specs["entity_source_grouped_bar"]["encoding"]
    assert {"x": "entity", "y": "count", "series": "metric"} == specs["source_cleaning_result_bar"]["encoding"]
    assert any(row["entity"] == "化学成分" and row["source"] == "BATMAN-TCM" and row["count"] == 24145 for row in entity_source_data)
    assert any(row["entity"] == "方剂" and row["metric"] == "最终结果" and row["count"] == 32646 for row in cleaning_data)
    assert_chart_specs(list(specs.values()))


def test_pathway_priority_handles_blank_pathway_names() -> None:
    predictor = HepatotoxicityPredictor.__new__(HepatotoxicityPredictor)
    pathways = pd.DataFrame(
        [
            {"Symbol": "CYP3A4", "PathwayName": "Drug metabolism"},
            {"Symbol": "ADD2", "PathwayName": np.nan},
            {"Symbol": "ATOH8", "PathwayName": ""},
            {"Symbol": "CYP2D6", "PathwayName": "Drug metabolism"},
        ]
    )

    ranked = predictor._add_pathway_priority(pathways)

    assert ranked["hit_count"].isna().sum() == 0
    blank_rows = ranked[ranked["Symbol"].isin(["ADD2", "ATOH8"])]
    assert blank_rows["hit_count"].tolist() == [0, 0]
    assert blank_rows["pathway_priority_score"].tolist() == [0.0, 0.0]


def test_json_safe_value_removes_non_finite_values() -> None:
    safe = HepatotoxicityPredictor._json_safe_value(
        {
            "nan": np.nan,
            "inf": float("inf"),
            "missing": pd.NA,
            "items": [1, np.float64("-inf"), {"score": np.float64(0.5)}],
        }
    )

    assert safe == {"nan": None, "inf": None, "missing": None, "items": [1, None, {"score": 0.5}]}
    json.dumps(safe, allow_nan=False)


def test_pathway_section_page_handles_blank_names(tmp_path) -> None:
    output_dir = tmp_path
    (output_dir / "00_output_file_manifest.csv").write_text(
        "FileName,RelativePath,FileType,Rows,Columns,Size_MB\n"
        "08_high_toxic_target_pathways.csv,08_high_toxic_target_pathways.csv,csv,3,5,0.001\n",
        encoding="utf-8",
    )
    (output_dir / "08_high_toxic_target_pathways.csv").write_text(
        "ChemicalName,GeneSymbol,GeneID,PathwayID,PathwayName\n"
        "Emodin,ADD2,,,\n"
        "Emodin,CYP3A4,1576,hsa00982,Drug metabolism\n"
        "Emodin,ATOH8,,,\n",
        encoding="utf-8",
    )
    predictor = HepatotoxicityPredictor.__new__(HepatotoxicityPredictor)

    page = predictor.get_output_section_page(
        query_type="formula",
        output_dir=output_dir,
        section_id="toxic_pathways",
        page=1,
        page_size=5,
    )

    assert page["section"]["row_count"] == 3
    assert page["section"]["sort"] == "pathway_priority_score"
    assert page["headers"] == ["Symbol", "ENTREZID", "Pathwayid", "PathwayName"]
    assert "GeneID" not in page["headers"]
    assert "GeneSymbol" not in page["headers"]
    assert "PathwayID" not in page["headers"]
    assert "ChemicalName" not in page["headers"]
    assert all(set(record) == {"Symbol", "ENTREZID", "Pathwayid", "PathwayName"} for record in page["records"])


def test_large_disease_section_streams_and_uses_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(predictor_module, "EXTERNAL_LARGE_CSV_THRESHOLD_BYTES", 1)
    output_dir = tmp_path
    (output_dir / "00_output_file_manifest.csv").write_text(
        "FileName,RelativePath,FileType,Rows,Columns,Size_MB\n"
        "10_high_toxic_target_disease_results.csv,10_high_toxic_target_disease_results.csv,csv,3,4,1.2\n",
        encoding="utf-8",
    )
    (output_dir / "10_high_toxic_target_disease_results.csv").write_text(
        "ChemicalName,Symbol,DiseaseID,DiseaseName\n"
        "A,CYP3A4,D1,Kidney injury\n"
        "B,CYP2E1,D2,Drug-induced liver injury\n"
        "C,ADD2,D3,\n",
        encoding="utf-8",
    )
    predictor = HepatotoxicityPredictor.__new__(HepatotoxicityPredictor)

    page = predictor.get_output_section_page(
        query_type="formula",
        output_dir=output_dir,
        section_id="toxic_diseases",
        page=1,
        page_size=2,
    )

    assert page["section"]["row_count"] == 3
    assert page["section"]["total_pages"] == 2
    assert page["records"][0]["DiseaseName"] == "Kidney injury"  # 相同成分数不按疾病关键词加权
    json.dumps(page, ensure_ascii=False, allow_nan=False)


def test_disease_section_page_rebuilds_target_disease_links(tmp_path) -> None:
    write_minimal_data_files(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "00_output_file_manifest.csv").write_text(
        "FileName,RelativePath,FileType,Rows,Columns,Size_MB\n"
        "07_high_toxic_chemical_targets.csv,07_high_toxic_chemical_targets.csv,csv,1,3,0.001\n"
        "10_high_toxic_target_disease_results.csv,10_high_toxic_target_disease_results.csv,csv,1,5,0.001\n",
        encoding="utf-8",
    )
    (output_dir / "07_high_toxic_chemical_targets.csv").write_text(
        "ChemicalName,GeneSymbol,GeneID\nEmodin,CYP3A4,1576\n",
        encoding="utf-8",
    )
    (output_dir / "10_high_toxic_target_disease_results.csv").write_text(
        "ChemicalName,GeneID,GeneSymbol,DiseaseName,DiseaseID\nEmodin,未提供,未提供,未提供,未提供\n",
        encoding="utf-8",
    )
    predictor = HepatotoxicityPredictor(root=tmp_path)

    page = predictor.get_output_section_page(
        query_type="formula",
        output_dir=output_dir,
        section_id="toxic_diseases",
        page=1,
        page_size=10,
    )

    assert page["headers"] == ["ChemicalName", "ENTREZID", "Symbol", "DiseaseName", "DiseaseID", "Raw_Record_Count"]
    assert "GeneID" not in page["headers"]
    assert "GeneSymbol" not in page["headers"]
    record = page["records"][0]
    assert record["ChemicalName"] == "Emodin"
    assert record["ENTREZID"] == "1576"
    assert record["Symbol"] == "CYP3A4"
    assert record["DiseaseName"] == "Drug-induced liver injury"
    assert record["DiseaseID"] == "MESH:D056486"


def test_disease_section_page_fills_chemical_name_from_compound_target(tmp_path) -> None:
    write_minimal_data_files(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "00_output_file_manifest.csv").write_text(
        "FileName,RelativePath,FileType,Rows,Columns,Size_MB\n"
        "07_high_toxic_chemical_targets.csv,07_high_toxic_chemical_targets.csv,csv,1,2,0.001\n"
        "10_high_toxic_target_disease_results.csv,10_high_toxic_target_disease_results.csv,csv,1,4,0.001\n",
        encoding="utf-8",
    )
    (output_dir / "07_high_toxic_chemical_targets.csv").write_text(
        "GeneSymbol,GeneID\nCYP3A4,1576\n",
        encoding="utf-8",
    )
    (output_dir / "10_high_toxic_target_disease_results.csv").write_text(
        "Symbol,GeneID,DiseaseName,DiseaseID\nCYP3A4,1576,未提供,未提供\n",
        encoding="utf-8",
    )
    predictor = HepatotoxicityPredictor(root=tmp_path)

    page = predictor.get_output_section_page(
        query_type="formula",
        output_dir=output_dir,
        section_id="toxic_diseases",
        page=1,
        page_size=10,
    )

    record = page["records"][0]
    assert page["headers"] == ["ChemicalName", "ENTREZID", "Symbol", "DiseaseName", "DiseaseID", "Raw_Record_Count"]
    assert record["ChemicalName"] == "Emodin"
    assert record["ENTREZID"] == "1576"
    assert record["Symbol"] == "CYP3A4"
    assert record["DiseaseName"] == "Drug-induced liver injury"
    assert record["DiseaseID"] == "MESH:D056486"


def test_disease_section_page_fills_entrez_from_target_disease_source(tmp_path) -> None:
    write_minimal_data_files(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "00_output_file_manifest.csv").write_text(
        "FileName,RelativePath,FileType,Rows,Columns,Size_MB\n"
        "07_high_toxic_chemical_targets.csv,07_high_toxic_chemical_targets.csv,csv,1,2,0.001\n"
        "10_high_toxic_target_disease_results.csv,10_high_toxic_target_disease_results.csv,csv,1,4,0.001\n",
        encoding="utf-8",
    )
    (output_dir / "07_high_toxic_chemical_targets.csv").write_text(
        "ChemicalName,GeneSymbol\nEmodin,CYP3A4\n",
        encoding="utf-8",
    )
    (output_dir / "10_high_toxic_target_disease_results.csv").write_text(
        "Symbol,GeneID,DiseaseName,DiseaseID\nCYP3A4,未提供,未提供,未提供\n",
        encoding="utf-8",
    )
    predictor = HepatotoxicityPredictor(root=tmp_path)

    page = predictor.get_output_section_page(
        query_type="formula",
        output_dir=output_dir,
        section_id="toxic_diseases",
        page=1,
        page_size=10,
    )

    record = page["records"][0]
    assert page["headers"] == ["ChemicalName", "ENTREZID", "Symbol", "DiseaseName", "DiseaseID", "Raw_Record_Count"]
    assert record["ChemicalName"] == "Emodin"
    assert record["ENTREZID"] == "1576"
    assert record["Symbol"] == "CYP3A4"
    assert record["DiseaseName"] == "Drug-induced liver injury"
    assert record["DiseaseID"] == "MESH:D056486"


def test_disease_section_page_fills_chemical_name_from_entrez_only_target(tmp_path) -> None:
    write_minimal_data_files(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "00_output_file_manifest.csv").write_text(
        "FileName,RelativePath,FileType,Rows,Columns,Size_MB\n"
        "07_high_toxic_chemical_targets.csv,07_high_toxic_chemical_targets.csv,csv,1,1,0.001\n"
        "10_high_toxic_target_disease_results.csv,10_high_toxic_target_disease_results.csv,csv,1,4,0.001\n",
        encoding="utf-8",
    )
    (output_dir / "07_high_toxic_chemical_targets.csv").write_text(
        "GeneID\n1576\n",
        encoding="utf-8",
    )
    (output_dir / "10_high_toxic_target_disease_results.csv").write_text(
        "GeneID,DiseaseName,DiseaseID\n1576,未提供,未提供\n",
        encoding="utf-8",
    )
    predictor = HepatotoxicityPredictor(root=tmp_path)

    page = predictor.get_output_section_page(
        query_type="formula",
        output_dir=output_dir,
        section_id="toxic_diseases",
        page=1,
        page_size=10,
    )

    record = page["records"][0]
    assert page["headers"] == ["ChemicalName", "ENTREZID", "Symbol", "DiseaseName", "DiseaseID", "Raw_Record_Count"]
    assert record["ChemicalName"] == "Emodin"
    assert record["ENTREZID"] == "1576"
    assert record["Symbol"] == "CYP3A4"
    assert record["DiseaseName"] == "Drug-induced liver injury"
    assert record["DiseaseID"] == "MESH:D056486"


def test_association_counts_use_unique_entities_and_keep_raw_frequency():
    targets = pd.DataFrame({"CID": [1, 1, 2], "ChemicalName": ["A", "A", "B"], "Symbol": ["T", "T", "T"]})
    assert HepatotoxicityPredictor._association_counts(targets, "Symbol", "compound")[0]["count"] == 2
    pathways = pd.DataFrame({"PathwayName": ["P", "P", "P"], "ENTREZID": [1, 1, 2]})
    assert HepatotoxicityPredictor._association_counts(pathways, "PathwayName", "target")[0]["count"] == 2
    diseases = targets.assign(DiseaseName="D", Raw_Record_Count=[2, 3, 4])
    counts = HepatotoxicityPredictor._association_counts(diseases, "DiseaseName", "compound")[0]
    assert counts["count"] == 2 and counts["raw_record_count"] == 9


def test_external_empty_assessment_keeps_exportable_headers_and_nulls(tmp_path, monkeypatch):
    from prediction_scripts.formula_livertox_pred import evaluate_compounds
    write_minimal_data_files(tmp_path)
    predictor = HepatotoxicityPredictor(tmp_path)
    config = predictor_module.EXTERNAL_PREDICTION_CONFIG["compound"]
    output = tmp_path / "output"
    output.mkdir()
    for _, filename in config["overview_files"]:
        pd.DataFrame(columns=["note"]).to_csv(output / filename, index=False)
    frame = evaluate_compounds(pd.DataFrame({"CID": [15945058], "ChemicalName": ["Emodin"], "Smiles": ["C"], "Reference_Match": [False], "Bioavailability_Ma": [.2]}))
    frame.to_csv(output / config["livertox_file"], index=False)
    for section, (_, filename) in config["tab_files"].items():
        (frame.iloc[:0] if section == "toxic_compounds" else pd.DataFrame(columns=["ChemicalName", "Symbol"])).to_csv(output / filename, index=False)
    monkeypatch.setattr(predictor, "_load_models", lambda: ())
    monkeypatch.setattr(predictor_module, "model_metadata", lambda _: {})
    result = predictor._build_external_prediction_response("compound", ["15945058"], output, config, job_id="empty")
    assert result["summary"]["assessment_status"] == "not_evaluable"
    assert all(value is None for value in result["probabilities"].values())
    assert result["compounds"][0]["Max_Tox_Prob"] is None
    assert all(row["row_count"] == 0 for row in result["sections"] if row["id"].startswith("toxic_"))
    for _, filename in config["tab_files"].values():
        assert len(pd.read_csv(output / filename).columns) > 0
    json.dumps(result, allow_nan=False)


def test_disease_raw_frequency_retains_source_duplicates(tmp_path):
    write_minimal_data_files(tmp_path)
    path = tmp_path / "data/08_target_disease.csv"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("CYP3A4,1576,Drug-induced liver injury,MESH:D056486\n")
    predictor = HepatotoxicityPredictor(tmp_path)
    targets = pd.DataFrame({"CID": [15945058, 15945058], "ChemicalName": ["Emodin", "Emodin alias"], "Symbol": ["CYP3A4", "CYP3A4"], "ENTREZID": [1576, 1576]})
    links = predictor._target_disease_frame_from_targets(targets)
    assert len(links) == 1
    assert links.iloc[0]["Raw_Record_Count"] == 2
    counts = predictor._association_counts(links, "DiseaseName", "compound")[0]
    assert counts["count"] == 1 and counts["raw_record_count"] == 2
