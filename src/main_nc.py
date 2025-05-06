import os
import os.path as osp
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import dgl
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GCN, GraphSAGE, GAT
import logging

from nc_dataset import NodeClassificationDataset
from nsg import NSG

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


def train(cfg, data: Data, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=5e-4)
    train_mask, val_mask = data.train_mask, data.val_mask
    for epoch in range(cfg.n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        # validate
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[val_mask] == data.y[val_mask]).sum()
        val_acc = int(correct) / int(val_mask.sum())
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')


@torch.no_grad
def test(data: Data, model):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Final Test Accuracy: {acc:.4f}')


@hydra.main(config_path='../configs', config_name="config_nc")
def main(cfg: DictConfig):
    ori_dir = hydra.utils.get_original_cwd()
    print(os.getcwd())
    print(os.getcwd())
    print(os.getcwd())
    print(os.getcwd())
    data_path = osp.join(ori_dir, '../data')
    verbose = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = NodeClassificationDataset(root=osp.join(data_path, cfg.dataset), feat_name=cfg.feat, verbose=verbose, device=device)
    data = dataset.graph
    # labels = dataset.labels
    # clip_feat = torch.load(osp.join(data_path, cfg.feat + '_feat.pt'))
    # if cfg.use_feature == 'text':
    #     clip_feat = clip_feat[:, :768]  # first 768 fea
    # g.ndata['feat'] = clip_feat
    # g.ndata['label'] = labels

    data = data.to(device)
    # g = dgl.remove_self_loop(g)
    # g, reverse_eids = to_bidirected_with_reverse_mapping(g)
    # g = g.to("cuda" if cfg.mode == "puregpu" else "cpu")
    # if cfg.dataset == 'books':
    #     print("using books")
    #     num_classes = len(torch.unique(labels))
    # else:
    #     num_classes = 12

    # splits = {}
    # splits['train_idx'] = ndata['train_mask'].nonzero()
    # splits['val_idx'] = ndata['val_mask'].nonzero()
    # splits['test_idx'] = ndata['test_mask'].nonzero()

    # num_nodes = g.num_nodes()
    # # create GraphSAGE model
    # in_size = g.ndata["feat"].shape[1]
    # out_size = num_classes
    # accs = []
    # for run in range(cfg.runs):
    #     log.info("Run {}/{}".format(run + 1, cfg.runs))

    if cfg.model == "SAGE":
        model = GraphSAGE(-1, cfg.hidden_dim, cfg.num_layers, dataset.n_classes, dropout=0.5)
    elif cfg.model == "GCN":
        model = GCN(-1, cfg.hidden_dim, cfg.num_layers, dataset.n_classes)
    elif cfg.model == "GAT":
        model = GAT(-1, cfg.hidden_dim, cfg.num_layers, dataset.n_classes, dropout=0.6, heads=8)
    elif cfg.model in ['HAN', 'HGT']:
        model = NSG(cfg.model, cfg.hidden_dim, [768], dataset.n_classes,device)

    model = model.to(device)
    train(cfg, data, model)

    test(data, model)


if __name__ == '__main__':
    main()
