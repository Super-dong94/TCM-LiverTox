from __future__ import annotations

from pathlib import Path

import numpy as np

from prediction_scripts.model_runtime import (
    load_model_bundles,
    predict_multimodel_pipeline,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = (
    ROOT / "models" / "01best_model_tuned.joblib",
    ROOT / "models" / "02best_model_tuned.joblib",
    ROOT / "models" / "03best_model_tuned.joblib",
)


def test_real_model_bundles_predict_urea() -> None:
    models = load_model_bundles(MODEL_PATHS)
    outputs = predict_multimodel_pipeline(["C(=O)(N)N"], models)

    assert len(outputs) == 6
    for index, values in enumerate(outputs):
        assert values.shape == (1,)
        if index % 2 == 0:
            assert int(values[0]) in {0, 1}
        else:
            assert np.isfinite(values[0])
            assert 0.0 <= float(values[0]) <= 1.0


def test_endpoint_boundaries_all_combinations_ties_and_empty():
    from itertools import product
    import pandas as pd
    from prediction_scripts.formula_livertox_pred import add_endpoint_scores, prediction_summary, evaluate_compounds
    thresholds = [.55, .96, .64]
    bits = list(product([0, 1], repeat=3))
    frame = pd.DataFrame([[t if bit else t - 1e-12 for t, bit in zip(thresholds, pattern)] for pattern in bits],
                         columns=["Pred_Cell_prob", "Pred_Animal_prob", "Pred_Clinical_prob"])
    scored = add_endpoint_scores(frame)
    assert scored["Hepatotoxicity_Score"].tolist() == [sum(pattern) for pattern in bits]
    assert scored["Endpoint_Combination"].nunique() == 8
    tied = add_endpoint_scores(pd.DataFrame({"Pred_Cell_prob": [.8, None], "Pred_Animal_prob": [.8, None], "Pred_Clinical_prob": [.2, None]}))
    assert tied.loc[0, "Pmax_Source"] == "cell;animal"
    assert pd.isna(tied.loc[1, "Max_Tox_Prob"])
    empty = evaluate_compounds(pd.DataFrame(columns=["CID", "Smiles"]))
    summary, probabilities, stats = prediction_summary(empty)
    assert summary["assessment_status"] == "not_evaluable"
    assert all(value is None for value in probabilities.values())
    assert len(stats["combination_counts"]) == 8
    assert sum(row["count"] for row in stats["score_counts"]) == 0


def test_standardization_and_unique_cids_reuse_inference(monkeypatch):
    import pandas as pd
    from prediction_scripts import formula_livertox_pred as core
    from prediction_scripts.model_runtime import standardize_smiles
    assert standardize_smiles("CC(=O)[O-].[Na+]") == "CC(=O)O"
    assert standardize_smiles("invalid") is None
    assert "@" in standardize_smiles("N[C@@H](C)C(=O)O")
    calls = []
    def predict(smiles, models):
        calls.append(smiles)
        return tuple(np.ones(len(smiles)) * value for value in [1, .6, 1, .97, 1, .7])
    monkeypatch.setattr(core, "predict_multimodel_pipeline", predict)
    models = [{"endpoint": endpoint, "threshold": threshold} for endpoint, threshold in zip(["cell", "animal", "clinical"], [.55, .96, .64])]
    frame = pd.DataFrame({"CID": [1, 2, 2, 3, 4], "Smiles": ["CC(=O)[O-].[Na+]", "CC(=O)O", "CC(=O)O", None, "C"],
                          "Reference_Match": [True, True, True, True, False]})
    result = core.evaluate_compounds(frame, models)
    assert len(result) == 4
    assert calls == [["CC(=O)O"]]
    assert result["Hepatotoxicity_Score"].notna().sum() == 2
    assert result["Max_Tox_Prob"].iloc[2:].isna().all()
    summary, _, _ = core.prediction_summary(result)
    assert summary["total_compounds"] == 4 and summary["candidate_count"] == 2


def test_nine_manuscript_compounds_with_real_models():
    import pandas as pd
    from prediction_scripts.formula_livertox_pred import evaluate_compounds, prediction_summary
    from prediction_scripts.model_runtime import model_metadata
    cids = [3220, 10168, 10639, 10208, 5280906, 107985, 821347, 161954, 11725801]
    reference = pd.read_csv(ROOT / "data/12_intoblood_ref.csv")
    models = load_model_bundles(MODEL_PATHS)
    scored = evaluate_compounds(reference[reference.CID.isin(cids)], models)
    summary, _, stats = prediction_summary(scored)
    assert summary["evaluated_count"] == 9
    assert [row["positive"] for row in stats["endpoint_counts"]] == [7, 9, 8]
    assert scored["Hepatotoxicity_Score"].value_counts().to_dict() == {3.0: 6, 2.0: 3}
    assert [info["calibration_method"] for info in model_metadata(models).values()] == ["none", "sigmoid", "none"]


def test_candidate_associations_do_not_expand_shared_structures():
    import pandas as pd
    from prediction_scripts.formula_livertox_pred import candidate_target_links
    candidates = pd.DataFrame({"CID": [1, 2], "Smiles": ["CC", "CC"], "ChemicalName": ["A", "B"], "Source_Herbs": ["H1", "H2"]})
    names = pd.DataFrame({"CID": [1, 2, 3], "ChemicalName": ["A", "B", "C"]})
    targets = pd.DataFrame({"ChemicalName": ["A", "B", "C"], "Symbol": ["T1", "T2", "T3"]})
    links = candidate_target_links(candidates, names, targets)
    assert set(links.CID) == {1, 2}
    assert set(links.Symbol) == {"T1", "T2"}
    assert dict(zip(links.CID, links.Source_Herbs)) == {1: "H1", 2: "H2"}


def test_unevaluated_csv_roundtrip_keeps_null_probabilities():
    import io
    import pandas as pd
    from prediction_scripts.formula_livertox_pred import evaluate_compounds, prediction_summary
    frame = evaluate_compounds(pd.DataFrame({"CID": [1], "Smiles": ["CC"], "Reference_Match": [False], "Bioavailability_Ma": [.2]}))
    restored = pd.read_csv(io.StringIO(frame.to_csv(index=False)))
    summary, probabilities, stats = prediction_summary(restored)
    assert summary["assessment_status"] == "not_evaluable" and summary["screened_out_count"] == 1
    assert summary["candidate_ratio"] is None and all(value is None for value in probabilities.values())
    assert all(row["positive"] == row["negative"] == 0 for row in stats["endpoint_counts"])
