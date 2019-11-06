import torch.nn as nn
from torch_geometric.utils import to_networkx
import torch
import networkx as nx
from utils import train_utils
import numpy as np

class ExplainModule:
    def __init__(self, model, data, feat, label, pred, train_idx, args, writer=None,
            print_training=True, graph_mode=False, graph_idx=False):
        self.model = model
        self.model.eval()
        self.feat = feat
        self.label = label
        self.pred = pred
        self.train_idx = train_idx
        self.n_hops = args.num_gc_layers
        self.graph_mode = graph_mode
        self.graph_idx = graph_idx
        self.args = args
        self.writer = writer
        self.print_training = print_training
        self.networkx_graph = to_networkx(data)
        self.mask = nn.Parameter(torch.ones((data.edge_index.size()[1], 1)))
        self.scheduler, self.optimizer = train_utils.build_optimizer(args, [self.mask])
        #self.representer()

    def _neighborhood(self, node_idx):
        bt = nx.bfs_tree(self.networkx_graph, source=node_idx, depth_limit=self.n_hops).edges()
        return bt

    def extract_neighborhood(self, node_idx, graph_idx=0):
        neighborhood = self._neighborhood(node_idx)
        neighbors = np.array([j for i, j in neighborhood])
        sub_feat = self.feat[graph_idx, neighbors]
        sub_label = self.label[graph_idx][neighbors]
        return sub_feat, sub_label, neighbors

    def explain(self, node_idx, graph_idx=0, graph_mode=False, unconstrained=False, model='exp'):
        if not graph_mode:
          sub_feat, sub_label, neighbors = self.extract_neighborhood(node_idx, graph_idx)
        x = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(sub_label, dtype=torch.long)
        pred_label = self.pred[graph_idx][neighbors]
