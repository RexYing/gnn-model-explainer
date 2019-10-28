import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv

class GCNNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
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
        self.concat = concat
        if self.concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        print ('GCNNet pred_input_dim:', self.pred_input_dim)

        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)

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
        self.convs.append(GCNConv(self.hidden_dim, embedding_dim))

        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims, 
                self.label_dim)

        print ('len(self.convs):', len(self.convs))

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim):
        pred_input_dim = pred_input_dim
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def forward(self, data):
        x, edge_index, batch = data.feat, data.edge_index, data.batch

        x_all = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
            x_all.append(x)
        x_tensor = torch.cat(x_all, dim=1)
        return self.pred_model(x_tensor)

    def loss(self, pred, label):
        pred = torch.transpose(pred, 0, 1)
        pred = pred.view((1, pred.shape[0], pred.shape[1]))
        label = label.view((1, label.shape[0]))
        # print(pred)
        return F.cross_entropy(pred, label, size_average=True)
