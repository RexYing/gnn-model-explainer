import math

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

import utils.train_utils as train_utils

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    return row_grad + col_grad


def preprocess_cg(cg):
    if use_cuda:
        preprocessed_cg_tensor = torch.from_numpy(cg).cuda()
    else:
        preprocessed_cg_tensor = torch.from_numpy(cg)

    preprocessed_cg_tensor.unsqueeze_(0)
    return Variable(preprocessed_cg_tensor, requires_grad=False)


def numpy_to_torch(img, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v

class Explainer:
    def __init__(self, model, adj, feat, label, pred, args):
        self.model = model
        self.model.eval()
        self.adj = adj
        self.feat = feat
        self.label = label
        self.pred = pred
        self.n_hops = args.num_gc_layers
        self.neighborhoods = self._neighborhoods()
        self.args = args

    def _neighborhoods(self):
        hop_adj = power_adj = self.adj
        for i in range(self.n_hops-1):
            power_adj = np.matmul(power_adj, self.adj)
            hop_adj = hop_adj + power_adj
            hop_adj = (hop_adj > 0).astype(int)
        return hop_adj


    def explain(self, node_idx, graph_idx=0):
        '''Explain a single node prediction
        '''
        print('node label: ', self.label[graph_idx][node_idx])
        neighbors_adj_row = self.neighborhoods[graph_idx][node_idx, :]
        # index of the query node in the new adj
        node_idx_new = sum(neighbors_adj_row[:node_idx])
        neighbors = np.nonzero(neighbors_adj_row)[0]
        print('neigh idx: ', node_idx, node_idx_new)
        sub_adj = self.adj[graph_idx][neighbors][:, neighbors]
        sub_adj = np.expand_dims(sub_adj, axis=0)
        sub_feat = self.feat[graph_idx, neighbors]
        sub_feat = np.expand_dims(sub_feat, axis=0)

        adj = torch.tensor(sub_adj, dtype=torch.float)
        x = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        #print('loaded pred: ', self.pred[graph_idx][node_idx])

        explainer = ExplainModule(adj, x, self.model)

        self.model.train()
        for epoch in range(self.args.num_epochs):
            self.mask.zero_grad()
            ypred = self.model(x, masked_adj)


class ExplainModule(nn.Module):
    def __init__(self, adj, x, model, node_idx, graph_idx=0):
        super(ExplainModule, self).__init__()
        self.adj = adj
        self.x = x
        self.model = model
        self.node_idx = node_idx
        self.graph_idx = graph_idx

        init_strategy='normal'
        self.mask = self.construct_edge_mask(adj.size()[-1], init_strategy=init_strategy)
        self.optimizer = train_utils.build_optimizer(self.args, [self.mask])

        ypred = self.model(x, adj)[graph_idx][node_idx]
        print('rerun pred: ', ypred)
        masked_adj = adj * self.mask
        ypred = self.model(x, masked_adj)
        print('init mask pred: ', ypred[graph_idx][node_idx])

    def construct_edge_mask(self, num_nodes, init_strategy='normal', const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == 'normal':
            with torch.no_grad():
                std = nn.init.calculate_gain('relu') * math.sqrt(2.0 / (num_nodes + num_nodes))
                mask.normal_(1.0, std)
                mask.clamp_(0.0, 1.0)
        elif init_strategy == 'const':
            nn.init.constant_(mask, const_val)

        #print(mask)
        return mask

    def forward(self, node_idx):
        masked_adj = adj * self.mask
        ypred = self.model(x, adj)[self.graph_idx][self.node_idx]
        node_pred = ypred[self.graph_idx, node_idx, :]
        print(node_pred)
        return nn.Softmax()(node_pred)

    def loss(self, pred, pred_label)：
        '''
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        '''
        loss = pred[pred_label]
