import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv, GAE
from sklearn.metrics import roc_auc_score, average_precision_score

# Load and split data
dataset = Planetoid(root="/tmp/Cora", name="Cora")
data = dataset[0]
transform = RandomLinkSplit(
    num_val=0.1, num_test=0.2, is_undirected=True, 
    add_negative_train_samples=True, split_labels=True
)
train_data, val_data, test_data = transform(data)

# Model
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

model = GAE(Encoder(dataset.num_features, out_channels=16))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    loss.backward()
    optimizer.step()
    return float(loss)

for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

# Evaluation
@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    pos_pred = model.decode(z, data.pos_edge_label_index).sigmoid()
    neg_pred = model.decode(z, data.neg_edge_label_index).sigmoid()
    y_true = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)]).cpu()
    y_scores = torch.cat([pos_pred, neg_pred]).cpu()
    return roc_auc_score(y_true, y_scores), average_precision_score(y_true, y_scores)

val_auc, val_ap = test(val_data)
test_auc, test_ap = test(test_data)
print(f"Val AUC-ROC: {val_auc:.4f}, Val AP: {val_ap:.4f}")
print(f"Test AUC-ROC: {test_auc:.4f}, Test AP: {test_ap:.4f}")