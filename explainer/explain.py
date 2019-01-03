import math

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
import tensorboardX.utils
import torch
from torch.autograd import Variable
import torch.nn as nn

import utils.io_utils as io_utils
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
    def __init__(self, model, adj, feat, label, pred, args, writer=None):
        self.model = model
        self.model.eval()
        self.adj = adj
        self.feat = feat
        self.label = label
        self.pred = pred
        self.n_hops = args.num_gc_layers
        self.neighborhoods = self._neighborhoods()
        self.args = args
        self.writer = writer

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
        sub_label = self.label[graph_idx][neighbors]
        sub_label = np.expand_dims(sub_label, axis=0)

        adj = torch.tensor(sub_adj, dtype=torch.float)
        x = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(sub_label, dtype=torch.long)
        #print('loaded pred: ', self.pred[graph_idx][node_idx])
        pred_label = np.argmax(self.pred[graph_idx][neighbors], axis=1)
        print('pred label: ', pred_label[node_idx_new])

        explainer = ExplainModule(adj, x, self.model, label, self.args, writer=self.writer)

        self.model.eval()
        explainer.train()
        for epoch in range(self.args.num_epochs):
            explainer.optimizer.zero_grad()
            ypred = explainer(node_idx_new)
            loss = explainer.loss(ypred, pred_label, node_idx_new, epoch)
            loss.backward()

            explainer.optimizer.step()
            if explainer.scheduler is not None:
                explainer.scheduler.step()

            mask_density = explainer.mask_density()
            print('epoch: ', epoch, '; loss: ', loss.item(),
                  '; mask density: ', mask_density.item(),
                  '; pred: ', ypred)

            if self.writer is not None:
                self.writer.add_scalar('mask/density', mask_density, epoch)
                self.writer.add_scalar('optimization/lr', explainer.optimizer.param_groups[0]['lr'], epoch)
                if epoch % 100 == 0:
                    explainer.log_mask(epoch)
                    explainer.log_masked_adj(epoch)
                    explainer.log_adj_grad(node_idx_new, pred_label, epoch)




class ExplainModule(nn.Module):
    def __init__(self, adj, x, model, label, args, graph_idx=0, writer=None, use_sigmoid=True):
        super(ExplainModule, self).__init__()
        self.adj = adj
        self.x = x
        self.model = model
        self.label = label
        self.graph_idx = graph_idx
        self.args = args
        self.writer = writer
        self.use_sigmoid = use_sigmoid

        init_strategy='normal'
        self.mask = self.construct_edge_mask(adj.size()[-1], init_strategy=init_strategy)
        self.scheduler, self.optimizer = train_utils.build_optimizer(args, [self.mask])

        self.coeffs = {'size': 0.5, 'grad': 0, 'lap': 1.0}

        # ypred = self.model(x, adj)[graph_idx][node_idx]
        # print('rerun pred: ', ypred)
        # masked_adj = adj * self.mask
        # ypred = self.model(x, masked_adj)
        # print('init mask pred: ', ypred[graph_idx][node_idx])

    def construct_edge_mask(self, num_nodes, init_strategy='normal', const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == 'normal':
            with torch.no_grad():
                std = nn.init.calculate_gain('relu') * math.sqrt(2.0 / (num_nodes + num_nodes))
                mask.normal_(1.0, std)
                #mask.clamp_(0.0, 1.0)
        elif init_strategy == 'const':
            nn.init.constant_(mask, const_val)

        #print(mask)
        return mask

    def _masked_adj(self):
        sym_mask = torch.sigmoid(self.mask) if self.use_sigmoid else self.mask
        sym_mask = (sym_mask + sym_mask.t()) / 2
        return self.adj * sym_mask

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj())
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum

    def forward(self, node_idx):
        self.masked_adj = self._masked_adj()
        ypred = self.model(self.x, self.masked_adj)
        node_pred = ypred[self.graph_idx, node_idx, :]
        return nn.Softmax()(node_pred)

    def adj_feat_grad(self, node_idx, pred_label_node):
        self.adj.requires_grad = True
        self.x.requires_grad = True
        ypred = self.model(self.x, self.adj)
        logit = nn.Softmax()(ypred[self.graph_idx, node_idx, pred_label_node])
        loss = -torch.log(logit)
        loss.backward()
        #return (self.adj.grad+self.adj.grad.permute(0, 2, 1)) / 2
        return self.adj.grad, self.x.grad

    def loss(self, pred, pred_label, node_idx, epoch):
        '''
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        '''
        pred_label_node = pred_label[node_idx]
        logit = pred[pred_label_node]
        pred_loss = -torch.log(logit)

        # size
        mask = torch.sigmoid(self.mask) if self.use_sigmoid else self.mask
        size_loss = self.coeffs['size'] * torch.mean(mask)

        # entropy

        # laplacian
        D = torch.diag(torch.sum(self.masked_adj[0], 0))
        L = D - self.masked_adj[self.graph_idx]
        pred_label_t = torch.tensor(pred_label, dtype=torch.float)
        lap_loss = self.coeffs['lap'] * (pred_label_t @ L @ pred_label_t) / self.adj.numel()

        # grad
        # adj
        adj_grad, x_grad = self.adj_feat_grad(node_idx, pred_label_node)[self.graph_idx]
        grad_loss = self.coeffs['grad'] * -torch.mean(torch.abs(adj_grad) * mask)
        # feat
        x_grad_sum = torch.sum(x_grad, 1)
        grad_feat_loss = self.coeffs['featgrad'] * -torch.mean(x_grad_sum * mask)

        loss = pred_loss + size_loss + grad_loss + lap_loss
        if self.writer is not None:
            self.writer.add_scalar('optimization/size_loss', size_loss, epoch)
            self.writer.add_scalar('optimization/grad_loss', grad_loss, epoch)
            self.writer.add_scalar('optimization/pred_loss', pred_loss, epoch)
            self.writer.add_scalar('optimization/lap_loss', lap_loss, epoch)
            self.writer.add_scalar('optimization/overall_loss', loss, epoch)
        return loss

    def log_mask(self, epoch):
        plt.switch_backend('agg')
        fig = plt.figure(figsize=(8,6), dpi=200)
        plt.imshow(self.mask.cpu().detach().numpy(), cmap=plt.get_cmap('BuPu'))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.writer.add_image('mask/mask_all', tensorboardX.utils.figure_to_image(fig), epoch)

        fig = plt.figure(figsize=(8,6), dpi=200)
        # use [0] to remove the batch dim
        plt.imshow(self.masked_adj[0].cpu().detach().numpy(), cmap=plt.get_cmap('BuPu'))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.writer.add_image('mask/adj', tensorboardX.utils.figure_to_image(fig), epoch)

    def log_adj_grad(self, node_idx, pred_label, epoch):
        adj_grad = torch.abs(self.adj_feat_grad(node_idx, pred_label[node_idx]))[self.graph_idx]
        io_utils.log_matrix(self.writer, adj_grad, 'grad/adj', epoch)
        #self.adj.requires_grad = False

    def log_masked_adj(self, epoch):
        # use [0] to remove the batch dim
        masked_adj = self.masked_adj[0].cpu().detach().numpy()
        num_nodes = self.adj.size()[-1]
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        weighted_edge_list = [(i, j, masked_adj[i, j]) for i in range(num_nodes) for j in range(num_nodes) if masked_adj[i,j] > 0.01]
        G.add_weighted_edges_from(weighted_edge_list)
        edge_colors = [G[i][j]['weight'] for (i,j) in G.edges()]

        plt.switch_backend('agg')
        fig = plt.figure(figsize=(8,6), dpi=200)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=True, font_size=6,
                node_color='#336699',
                edge_color=edge_colors, edge_cmap=plt.get_cmap('Greys'), edge_vmin=0.0, edge_vmax=1.0,
                width=0.5, node_size=100,
                alpha=0.7)
        fig.axes[0].xaxis.set_visible(False)
        fig.canvas.draw()
        self.writer.add_image('mask/graph', tensorboardX.utils.figure_to_image(fig), epoch)
