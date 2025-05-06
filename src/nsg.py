import torch
import torch.nn as nn
from torch.nn import Module
from torch_geometric.data import Data, HeteroData
from itertools import combinations

from hgns import HAN, HGT


class NSG(Module):
    def __init__(self, model_name, hidden_dim, fea_split, out_channels, device):
        super().__init__()
        self.fea_split = fea_split = [0] + fea_split
        self.n_type = n_type = len(fea_split)
        self.tp_list = tp_list = [str(i) for i in range(n_type)]
        dumb = HeteroData()
        for tp in tp_list:
            dumb[tp].x = None
            dumb[(tp, tp)].edge_index = None
        for tp0, tp1 in combinations(tp_list, 2):
            dumb[(tp0, tp1)].edge_index = None
            dumb[(tp1, tp0)].edge_index = None
        metadata = dumb.metadata()
        if model_name == "HAN":
            self.encoder = HAN(-1, hidden_dim, hidden_dim, metadata)
        elif model_name == "HGT":
            self.encoder = HGT(hidden_dim, hidden_dim, metadata)
        self.fc = nn.LazyLinear(out_channels)
        self.device = device

    def forward(self, x, edge_index):
        '''Given a homogeneous graph data'''
        n_nodes = x.size(0)
        hdata = HeteroData()
        fea_split, n_type, tp_list = self.fea_split, self.n_type, self.tp_list
        for i, tp in enumerate(tp_list):
            hdata[tp].x = x[:, fea_split[i] : (fea_split[i + 1] if i + 1 < n_type else -1)]
            hdata[(tp, tp)].edge_index = edge_index.clone()
        '''Expert 0'''
        for tp0, tp1 in combinations(tp_list, 2):
            hdata[(tp0, tp1)].edge_index = torch.tensor([list(range(n_nodes)) for _ in range(2)]).long().to(self.device)
            hdata[(tp1, tp0)].edge_index = torch.tensor([list(range(n_nodes)) for _ in range(2)]).long().to(self.device)
        out = self.encoder(hdata.x_dict, hdata.edge_index_dict)
        out = self.fc(torch.cat([out[tp] for tp in tp_list], dim=1))
        return out
