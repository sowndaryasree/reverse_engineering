"""Loading the trained artifacts and running predictions (torch-dependent code)."""
import json
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).parent
ART = BASE / "artifacts"
REQUIRED = ["config.json", "metrics.json", "gnn_weights.pt", "sample_graphs.pt"]
GNN_NAME = "GraphSAGE GNN (uses graph)"
BASE_NAME = "Baseline (mean features, no graph)"


def missing_files():
    files = REQUIRED + ["confusion_matrix.png", "early_detection.png"]
    return [f for f in files if not (ART / f).exists()] + (["model_def.py"] if not (BASE / "model_def.py").exists() else [])


def load_config():
    return json.loads((ART / "config.json").read_text())


def load_metrics():
    return json.loads((ART / "metrics.json").read_text())


def load_model(cfg):
    """Build the SAME architecture used in Colab (model_def.py) and load the trained weights."""
    from model_def import GraphGNN
    model = GraphGNN(cfg["in_dim"], cfg["hidden"], cfg["num_classes"], cfg["conv"], cfg["dropout"])
    state = torch.load(ART / "gnn_weights.pt", map_location="cpu")
    model.load_state_dict(state)  # strict: fails loudly if shapes do not match
    model.eval()
    return model


def load_samples():
    samples = torch.load(ART / "sample_graphs.pt", map_location="cpu")
    out = []
    for s in samples:
        out.append({"x": s["x"].numpy(), "edge_index": s["edge_index"].numpy(),
                    "y": int(s["y"]), "root": int(s["root"])})
    return out


@torch.no_grad()
def predict_proba(model, x, edge_index):
    """x: (n, in_dim) numpy, edge_index: (2, E) numpy -> class probabilities (numpy)."""
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    ei = torch.from_numpy(np.asarray(edge_index, dtype=np.int64))
    batch = torch.zeros(xt.size(0), dtype=torch.long)
    out = model(xt, ei, batch)
    return torch.softmax(out, dim=1)[0].numpy()
