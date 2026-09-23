from __future__ import annotations

import pandas as pd

from prediction_scripts.intoblood_pred import match_intoblood_reference


def test_matches_by_cid_name_and_smiles_with_fixed_priority() -> None:
    reference = pd.DataFrame(
        [
            {
                "CID": 1,
                "ChemicalName": "Alpha",
                "Smiles": "C",
                "Bioavailability_Ma": 0.91,
                "Reference_Match": True,
                "IntoBlood": 1,
                "IntoBlood_Reason": "cid_hit",
            },
            {
                "CID": 2,
                "ChemicalName": "Beta",
                "Smiles": "CC",
                "Bioavailability_Ma": 0.12,
                "Reference_Match": False,
                "IntoBlood": 0,
                "IntoBlood_Reason": "name_hit",
            },
        ]
    )
    query = pd.DataFrame(
        [
            {"CID": "1.0", "ChemicalName": "Beta", "Smiles": "CC"},
            {"CID": 999, "ChemicalName": " beta ", "Smiles": "CCC"},
            {"CID": 998, "ChemicalName": "Unknown", "Smiles": " C "},
            {"CID": 997, "ChemicalName": "Missing", "Smiles": "N"},
        ]
    )

    result = match_intoblood_reference(query, reference)

    assert result["IntoBlood"].iloc[:3].tolist() == [1, 0, 1]
    assert pd.isna(result.loc[3, "IntoBlood"])
    assert result["Precomputed_Match"].tolist() == [True, True, True, False]
    assert result["IntoBlood_Reason"].tolist() == [
        "reference_match",
        "admet_below_0.3",
        "reference_match",
        "missing_admet",
    ]
    assert result["Reference_Match"].tolist() == [True, False, True, False]
    assert pd.isna(result.loc[3, "Bioavailability_Ma"])


def test_blood_rules_missing_structure_and_boundaries():
    from prediction_scripts.intoblood_pred import assess_blood_exposure
    frame = pd.DataFrame({"Smiles": ["C", "CC", "CCC", "invalid", "O", ""],
                          "Reference_Match": [True, False, False, True, False, True],
                          "Bioavailability_Ma": [None, .3, .3 - 1e-12, None, None, .8]})
    result = assess_blood_exposure(frame)
    assert result["IntoBlood"].iloc[:3].tolist() == [1, 1, 0]
    assert result["IntoBlood"].iloc[3:].isna().all()
    assert result["IntoBlood_Reason"].tolist() == ["reference_match", "admet_ge_0.3", "admet_below_0.3", "invalid_structure", "missing_admet", "invalid_structure"]


def test_full_blood_reference_matches_manuscript_counts():
    from pathlib import Path
    from prediction_scripts.intoblood_pred import assess_blood_exposure
    frame = pd.read_csv(Path(__file__).resolve().parents[1] / "data/12_intoblood_ref.csv")
    result = assess_blood_exposure(frame)
    assert len(result) == 37535
    assert result["IntoBlood_Reason"].eq("reference_match").sum() == 1523
    assert result["IntoBlood_Reason"].eq("admet_ge_0.3").sum() == 31506
    assert result["IntoBlood"].eq(1).sum() == 33029
