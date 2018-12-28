import math

import torch
from torch.autograd import Variable
import torch.nn as nn

import numpy as np

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

    def _neighborhoods(self):
        hop_adj = power_adj = self.adj
        for i in range(self.n_hops-1):
            power_adj = np.matmul(power_adj, self.adj)
            hop_adj = hop_adj + power_adj
            hop_adj = (hop_adj > 0).astype(int)
        return hop_adj

    def construct_edge_mask(self, num_nodes, init_strategy='normal'):
        mask = nn.Parameter(torch.DoubleTensor(num_nodes, num_nodes))
        if init_strategy == 'normal':
            with torch.no_grad():
                std = nn.init.calculate_gain('relu') * math.sqrt(2.0 / (num_nodes + num_nodes))
                mask.normal_(1.0, std)
                mask.clamp_(0.0, 1.0)
        elif init_strategy == 'const':
            nn.init.constant_(mask, 1.0)

        print(mask)
        return mask

    def explain(self, node_idx, graph_idx=0):
        '''Explain a single node prediction
        '''
        print('node label: ', self.label[graph_idx][node_idx])
        neighbors_adj_row = self.neighborhoods[graph_idx][node_idx, :]
        node_idx_new = sum(neighbors_adj_row[:node_idx])
        neighbors = np.nonzero(neighbors_adj_row)[0]
        print('neigh idx: ', node_idx, node_idx_new)
        sub_adj = self.adj[graph_idx][neighbors][:, neighbors]
        sub_adj = np.expand_dims(sub_adj, axis=0)
        sub_feat = self.feat[graph_idx, neighbors]
        sub_feat = np.expand_dims(sub_feat, axis=0)

        adj = torch.tensor(sub_adj, dtype=torch.float)
        x = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        ypred = self.model(x, adj)[graph_idx][node_idx_new]
        #print('rerun pred: ', ypred)
        #print('loaded pred: ', self.pred[graph_idx][node_idx])

        self.construct_edge_mask(adj.size()[-1])
