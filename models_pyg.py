import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv

class GCNNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, add_self=False, args=None):
        super(GCNNet, self).__init__()
        self.input_dim = input_dim
        print ('GCNNet input_dim:', self.input_dim)
        self.hidden_dim = hidden_dim
        print ('GCNNet hidden_dim:', self.hidden_dim)
        self.label_dim = label_dim
        print ('GCNNet label_dim:', self.label_dim)
        self.num_layers = num_layers
        print ('GCNNet num_layers:', self.num_layers)

        # self.concat = concat
        # self.bn = bn
        # self.add_self = add_self
        self.args = args
        self.dropout = dropout
        self.act = F.relu

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(self.input_dim, self.hidden_dim))
        for layer in range(self.num_layers - 2):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
        self.convs.append(GCNConv(self.hidden_dim, self.label_dim))
        print ('len(self.convs):', len(self.convs))

    def forward(self, data):
        x, edge_index, batch = data.feat, data.edge_index, data.batch

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
