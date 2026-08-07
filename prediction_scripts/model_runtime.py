from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import warnings

import joblib
import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, GINEConv, global_max_pool, global_mean_pool


class FingerprintMLP(nn.Module):
    def __init__(self, input_size: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, max(32, hidden // 2)),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(max(32, hidden // 2), 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class GCNModel(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.input_layer = nn.Linear(9, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, 2),
        )

    def forward(self, data: Data) -> torch.Tensor:
        values = F.relu(self.input_layer(data.x))
        values = F.relu(self.conv1(values, data.edge_index))
        values = F.relu(self.conv2(values, data.edge_index))
        pooled = torch.cat(
            [global_mean_pool(values, data.batch), global_max_pool(values, data.batch)],
            dim=1,
        )
        return self.head(pooled)


class GINEModel(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.input_layer = nn.Linear(9, hidden)
        update1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        update2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.conv1 = GINEConv(update1, edge_dim=4)
        self.conv2 = GINEConv(update2, edge_dim=4)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, 2),
        )

    def forward(self, data: Data) -> torch.Tensor:
        values = F.relu(self.input_layer(data.x))
        values = F.relu(self.conv1(values, data.edge_index, data.edge_attr))
        values = F.relu(self.conv2(values, data.edge_index, data.edge_attr))
        pooled = torch.cat(
            [global_mean_pool(values, data.batch), global_max_pool(values, data.batch)],
            dim=1,
        )
        return self.head(pooled)


def load_model_bundles(
    model_paths: Iterable[str | Path],
    expected_endpoints: Iterable[str] = ("cell", "animal", "clinical"),
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bundles = []
    for path, endpoint in zip(model_paths, expected_endpoints):
        bundle = joblib.load(Path(path))
        if not isinstance(bundle, dict) or bundle.get("endpoint") != endpoint:
            raise ValueError(f"模型文件端点不匹配: {path}，期望 {endpoint}。")
        bundles.append(bundle)
    if len(bundles) != 3:
        raise ValueError("必须提供细胞、动物、临床三个模型文件。")
    return tuple(bundles)  # type: ignore[return-value]


def _vector_features(smiles_list: list[str], bundle: dict[str, Any]) -> np.ndarray:
    schema = bundle["feature_schema"]
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(schema["morgan_radius"]),
        fpSize=int(schema["morgan_bits"]),
    )
    descriptor_map = dict(Descriptors._descList)
    descriptor_functions = [descriptor_map.get(name) for name in bundle["descriptor_names"]]
    rows = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无法解析 SMILES: {smiles}")
        morgan = generator.GetFingerprintAsNumPy(mol).astype(np.float32, copy=False)
        maccs = np.asarray(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
        descriptors = []
        for function in descriptor_functions:
            try:
                value = float(function(mol)) if function is not None else np.nan
            except Exception:
                value = np.nan
            descriptors.append(value)
        rows.append(
            np.concatenate([morgan, maccs, np.asarray(descriptors, dtype=np.float32)])
        )
    matrix = np.asarray(rows, dtype=np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    expected_count = int(schema["feature_count"])
    if matrix.shape[1] != expected_count:
        raise ValueError(f"模型需要 {expected_count} 个特征，实际生成 {matrix.shape[1]} 个。")
    return matrix


def _smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smiles}")
    node_values = [
        [
            atom.GetAtomicNum() / 100.0,
            atom.GetTotalDegree() / 6.0,
            atom.GetFormalCharge() / 5.0,
            atom.GetTotalNumHs() / 4.0,
            float(atom.GetIsAromatic()),
            atom.GetMass() / 200.0,
            float(atom.IsInRing()),
            int(atom.GetChiralTag()) / 4.0,
            int(atom.GetHybridization()) / 8.0,
        ]
        for atom in mol.GetAtoms()
    ]
    edge_indices: list[list[int]] = []
    edge_values: list[list[float]] = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_value = [
            float(bond.GetBondTypeAsDouble()) / 3.0,
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            int(bond.GetStereo()) / 6.0,
        ]
        edge_indices.extend([[begin, end], [end, begin]])
        edge_values.extend([bond_value, bond_value])
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_values, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float32)
    return Data(
        x=torch.tensor(node_values, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


def _runtime_neural_model(entry: dict[str, Any]) -> nn.Module:
    model = entry.get("_runtime_model")
    if model is not None:
        return model
    hidden = int(entry["parameters"]["hidden"])
    if entry["name"] == "MLP_PyTorch":
        model = FingerprintMLP(int(entry["input_size"]), hidden)
    elif entry["name"] == "GCN":
        model = GCNModel(hidden)
    elif entry["name"] == "GINE":
        model = GINEModel(hidden)
    else:
        raise ValueError(f"不支持的神经网络模型: {entry['name']}")
    model.load_state_dict(entry["state_dict"])
    model.eval()
    entry["_runtime_model"] = model
    return model


def _neural_raw_score(
    entry: dict[str, Any],
    vector_values: np.ndarray,
    smiles_list: list[str],
) -> np.ndarray:
    model = _runtime_neural_model(entry)
    with torch.inference_mode():
        if entry["name"] == "MLP_PyTorch":
            values = np.nan_to_num(
                vector_values,
                nan=0.0,
                posinf=1e6,
                neginf=-1e6,
            )
            values = entry["scaler"].transform(values).astype(np.float32)
            logits = model(torch.tensor(values, dtype=torch.float32))
        else:
            batch = Batch.from_data_list([_smiles_to_graph(smiles) for smiles in smiles_list])
            logits = model(batch)
        return torch.softmax(logits, dim=1)[:, 1].cpu().numpy().astype(float)


def _apply_calibrator(
    raw_scores: np.ndarray,
    calibrator: Any,
    method: str,
) -> np.ndarray:
    raw_scores = np.asarray(raw_scores, dtype=float)
    if method == "none":
        if np.nanmin(raw_scores) < 0 or np.nanmax(raw_scores) > 1:
            return scipy.special.expit(np.clip(raw_scores, -30, 30))
        return np.clip(raw_scores, 1e-6, 1 - 1e-6)
    if method == "sigmoid":
        values = calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
    else:
        values = calibrator.predict(raw_scores)
    return np.clip(values, 1e-6, 1 - 1e-6)


def _entry_probability(
    entry: dict[str, Any],
    vector_values: np.ndarray,
    smiles_list: list[str],
) -> np.ndarray:
    if entry["kind"] == "vector":
        model = entry["model"]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            if entry["name"] == "SVM":
                raw_scores = model.decision_function(vector_values)
            else:
                raw_scores = model.predict_proba(vector_values)[:, 1]
    else:
        raw_scores = _neural_raw_score(entry, vector_values, smiles_list)
    return _apply_calibrator(
        np.asarray(raw_scores, dtype=float),
        entry["calibrator"],
        entry["calibration_method"],
    )


def predict_model_bundle(
    bundle: dict[str, Any],
    smiles_list: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    if not smiles_list:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    vector_values = _vector_features(smiles_list, bundle)
    mode = bundle["mode"]
    if mode == "base":
        probabilities = _entry_probability(bundle["entry"], vector_values, smiles_list)
    else:
        base_probabilities = np.column_stack(
            [
                _entry_probability(entry, vector_values, smiles_list)
                for entry in bundle["entries"].values()
            ]
        )
        if mode == "soft_voting":
            raw_scores = base_probabilities.mean(axis=1)
        elif mode == "stacking":
            raw_scores = bundle["meta_model"].predict_proba(base_probabilities)[:, 1]
        else:
            raise ValueError(f"不支持的模型包模式: {mode}")
        probabilities = _apply_calibrator(
            raw_scores,
            bundle["ensemble_calibrator"],
            bundle["ensemble_calibration_method"],
        )
    threshold = float(bundle["threshold"])
    predictions = (probabilities >= threshold).astype(int)
    return predictions, np.asarray(probabilities, dtype=float)


def predict_multimodel_pipeline(
    smiles_list: list[str],
    models: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cell, animal, clinical = models
    pred_cell, prob_cell = predict_model_bundle(cell, smiles_list)
    pred_animal, prob_animal = predict_model_bundle(animal, smiles_list)
    pred_clinical, prob_clinical = predict_model_bundle(clinical, smiles_list)
    return (
        pred_cell,
        prob_cell,
        pred_animal,
        prob_animal,
        pred_clinical,
        prob_clinical,
    )
