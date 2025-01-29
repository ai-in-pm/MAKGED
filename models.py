import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from config import GCN_HIDDEN_CHANNELS, GCN_NUM_LAYERS

class GCNEncoder(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, GCN_HIDDEN_CHANNELS))
        
        for _ in range(GCN_NUM_LAYERS - 2):
            self.convs.append(GCNConv(GCN_HIDDEN_CHANNELS, GCN_HIDDEN_CHANNELS))
        
        self.convs.append(GCNConv(GCN_HIDDEN_CHANNELS, GCN_HIDDEN_CHANNELS))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x
