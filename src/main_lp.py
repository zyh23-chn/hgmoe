import os
import os.path as osp
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import dgl
from torch_geometric.data import Data
from torch_geometric.nn import GCN, GraphSAGE, GAT, GAE
import logging

from lp_dataset import LinkPredictionDataset

# logging.basicConfig(level=logging.DEBUG)
# log = logging.getLogger(__name__)


def train(cfg, data: Data, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=5e-4)
    for epoch in range(cfg.n_epochs):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.pos_edge_label_index)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")


@torch.no_grad
def test(data: Data, model):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    pos_pred = model.decode(z, data.pos_edge_label_index).sigmoid()
    neg_pred = model.decode(z, data.neg_edge_label_index).sigmoid()

    # predictions = torch.cat([pos_pred, neg_pred])
    # _, indices = torch.sort(predictions, descending=True)
    # ranks = torch.nonzero(indices < pos_pred.size(0)).view(-1) + 1
    # # Compute MRR
    # mrr = torch.mean(1.0 / ranks.float())
    # # Compute Hits@K
    # hits = float(ranks.le(k).sum()) / pos_pred.size(0)

    y_true = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)]).cpu()
    y_scores = torch.cat([pos_pred, neg_pred]).cpu()
    return roc_auc_score(y_true, y_scores), average_precision_score(y_true, y_scores)


@hydra.main(config_path='../configs', config_name="config_lp")
def main(cfg: DictConfig):
    ori_dir = hydra.utils.get_original_cwd()
    print(os.getcwd())
    print(os.getcwd())
    print(os.getcwd())
    print(os.getcwd())
    data_path = osp.join(ori_dir, '../data')
    verbose = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = LinkPredictionDataset(root=osp.join(data_path, cfg.dataset), feat_name=cfg.feat, verbose=verbose, device=device)
    train_g, val_g, test_g = dataset.train_g.to(device), dataset.val_g.to(device), dataset.test_g.to(device)

    if cfg.model == "SAGE":
        model = GraphSAGE(-1, cfg.hidden_dim, cfg.num_layers, cfg.out_channels, dropout=0.5)
    elif cfg.model == "GCN":
        model = GCN(-1, cfg.hidden_dim, cfg.num_layers, cfg.out_channels)
    elif cfg.model == "GAT":
        model = GAT(-1, cfg.hidden_dim, cfg.num_layers, cfg.out_channels, dropout=0.6, heads=8)
    model = GAE(model).to(device)
    train(cfg, train_g, model)

    val_auc, val_ap = test(test_g, model)
    test_auc, test_ap = test(val_g, model)
    print(f"Val AUC-ROC: {val_auc:.4f}, Val AP: {val_ap:.4f}")
    print(f"Test AUC-ROC: {test_auc:.4f}, Test AP: {test_ap:.4f}")


if __name__ == '__main__':
    main()
