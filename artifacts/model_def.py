# CELL 2: model definitions. The Streamlit app will import THIS SAME FILE (so architectures always match).
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, global_mean_pool


class GraphGNN(nn.Module):
    """2-layer GraphSAGE (or GCN) -> mean pooling over the graph -> linear classifier."""
    def __init__(self, in_dim, hidden=64, num_classes=2, conv="sage", dropout=0.5):
        super().__init__()
        Conv = SAGEConv if conv == "sage" else GCNConv
        self.conv1 = Conv(in_dim, hidden)
        self.conv2 = Conv(hidden, hidden)
        self.lin = nn.Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.dropout(x, self.dropout, self.training)
        return self.lin(x)


class MeanFeatureMLP(nn.Module):
    """BASELINE: averages node features of the graph and classifies. Ignores edges (no graph structure)."""
    def __init__(self, in_dim, hidden=64, num_classes=2, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, self.training)
        return self.fc2(x)
