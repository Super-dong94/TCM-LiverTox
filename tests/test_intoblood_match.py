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

    assert result["IntoBlood"].tolist() == [1, 0, 1, 0]
    assert result["IntoBlood_Reason"].tolist() == [
        "cid_hit",
        "name_hit",
        "cid_hit",
        "not_in_reference",
    ]
    assert result["Reference_Match"].tolist() == [True, False, True, False]
    assert pd.isna(result.loc[3, "Bioavailability_Ma"])
