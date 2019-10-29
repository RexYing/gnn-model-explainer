import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

class GCNNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=False, bn=True, dropout=0.0, add_self=False, args=None):
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
        self.convs.append(self.build_conv_model(self.input_dim, self.hidden_dim))
        for layer in range(self.num_layers - 2):
            self.convs.append(self.build_conv_model(self.hidden_dim, self.hidden_dim))
        self.convs.append(self.build_conv_model(self.hidden_dim, embedding_dim))

        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims, 
                self.label_dim)

        print ('len(self.convs):', len(self.convs))

    def build_conv_model(self, input_dim, output_dim):
        args = self.args
        if args.method == 'base': # sage with add agg
            conv_model = MyConv(input_dim, output_dim)
        elif args.method == 'gcn':
            conv_model = pyg_nn.GCNConv(input_dim, output_dim)
        elif args.method == 'gin':
            conv_model = pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, output_dim),
                                  nn.ReLU(), nn.Linear(output_dim, output_dim)))
        return conv_model

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
        if self.concat:
            x_tensor = torch.cat(x_all, dim=1)
        else:
            x_tensor = x
        return self.pred_model(x_tensor)

    def loss(self, pred, label):
        pred = torch.transpose(pred, 0, 1)
        pred = pred.view((1, pred.shape[0], pred.shape[1]))
        label = label.view((1, label.shape[0]))
        # print(pred)
        return F.cross_entropy(pred, label, size_average=True)

class MyConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MyConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_update = nn.Linear(out_channels + in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        # Transform node feature matrix.
        #self_x = self.lin_self(x)

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j, edge_index, size):
        # Compute messages
        # x_j has shape [E, out_channels]
        return self.lin(x_j)

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        out = torch.cat((aggr_out, x), dim=1)
        out = self.lin_update(out)
        return out

