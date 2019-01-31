import math
import time

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import seaborn as sns
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
    def __init__(self, model, adj, feat, label, pred, train_idx, args, writer=None,
            print_training=True):
        self.model = model
        self.model.eval()
        self.adj = adj
        self.feat = feat
        self.label = label
        self.pred = pred
        self.train_idx = train_idx
        self.n_hops = args.num_gc_layers
        self.neighborhoods = self._neighborhoods()
        self.args = args
        self.writer = writer
        self.print_training = print_training

        self.representer()

    def _neighborhoods(self):
        hop_adj = power_adj = self.adj
        for i in range(self.n_hops-1):
            power_adj = np.matmul(power_adj, self.adj)
            hop_adj = hop_adj + power_adj
            hop_adj = (hop_adj > 0).astype(int)
        return hop_adj

    def representer(self):
        self.model.train()
        self.model.zero_grad()
        adj = torch.tensor(self.adj, dtype=torch.float)
        x = torch.tensor(self.feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(self.label, dtype=torch.long)
        if self.args.gpu:
            adj, x, label = adj.cuda(), x.cuda(), label.cuda()

        preds = self.model(x, adj)
        preds.retain_grad()
        self.embedding = self.model.embedding_tensor
        loss = self.model.loss(preds, label)
        loss.backward()
        self.preds_grad = preds.grad
        pred_idx = np.expand_dims(np.argmax(self.pred, axis=2), axis=2)
        pred_idx = torch.LongTensor(pred_idx)
        if self.args.gpu:
            pred_idx = pred_idx.cuda()
        #self.alpha = -self.preds_grad.gather(dim=2,
        #        index=pred_idx).squeeze(dim=2)
        self.alpha = self.preds_grad

    def extract_neighborhood(self, node_idx, graph_idx=0):
        neighbors_adj_row = self.neighborhoods[graph_idx][node_idx, :]
        # index of the query node in the new adj
        node_idx_new = sum(neighbors_adj_row[:node_idx])
        neighbors = np.nonzero(neighbors_adj_row)[0]
        sub_adj = self.adj[graph_idx][neighbors][:, neighbors]
        sub_feat = self.feat[graph_idx, neighbors]
        sub_label = self.label[graph_idx][neighbors]
        return node_idx_new, sub_adj, sub_feat, sub_label, neighbors

    def explain(self, node_idx, graph_idx=0, unconstrained=False):
        '''Explain a single node prediction
        '''
        print('node label: ', self.label[graph_idx][node_idx])
        # index of the query node in the new adj
        node_idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(node_idx, graph_idx)
        sub_adj = np.expand_dims(sub_adj, axis=0)
        sub_feat = np.expand_dims(sub_feat, axis=0)
        sub_label = np.expand_dims(sub_label, axis=0)
        print('neigh graph idx: ', node_idx, node_idx_new)

        adj = torch.tensor(sub_adj, dtype=torch.float)
        x = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(sub_label, dtype=torch.long)
        #print('loaded pred: ', self.pred[graph_idx][node_idx])
        pred_label = np.argmax(self.pred[graph_idx][neighbors], axis=1)
        print('pred label: ', pred_label[node_idx_new])

        #f_test = self.embedding[graph_idx, node_idx, :]
        #f_idx = self.embedding[graph_idx, self.train_idx, :]
        #alpha = self.alpha[graph_idx, self.train_idx, pred_label[node_idx_new]]
        #sim_val = f_idx @ f_test
        #rep_val = sim_val * alpha
        #if self.writer is not None:
        #    self.log_representer(rep_val, sim_val, alpha)

        explainer = ExplainModule(adj, x, self.model, label, self.args, writer=self.writer)
        if self.args.gpu:
            explainer = explainer.cuda()

        self.model.eval()
        explainer.train()
        begin_time = time.time()
        for epoch in range(self.args.num_epochs):
            explainer.zero_grad()
            explainer.optimizer.zero_grad()
            ypred = explainer(node_idx_new, unconstrained=unconstrained)
            loss = explainer.loss(ypred, pred_label, node_idx_new, epoch)
            loss.backward()

            explainer.optimizer.step()
            if explainer.scheduler is not None:
                explainer.scheduler.step()

            mask_density = explainer.mask_density()
            if self.print_training:
                print('epoch: ', epoch, '; loss: ', loss.item(),
                      '; mask density: ', mask_density.item(),
                      '; pred: ', ypred)

            if self.writer is not None:
                self.writer.add_scalar('mask/density', mask_density, epoch)
                self.writer.add_scalar('optimization/lr', explainer.optimizer.param_groups[0]['lr'], epoch)
                if epoch % 100 == 0:
                    explainer.log_mask(epoch)
                    explainer.log_masked_adj(node_idx_new, epoch, label=sub_label.squeeze())
                    explainer.log_adj_grad(node_idx_new, pred_label, epoch)

        print('finished training in ', time.time() - begin_time)
        masked_adj = explainer.masked_adj[0].cpu().detach().numpy()
        return masked_adj

    def remove_low_weight_edges(self):
        d

    def align(self, ref_feat, ref_adj, ref_node_idx, curr_feat, curr_adj, curr_node_idx, args):
        ref_adj = torch.FloatTensor(ref_adj)
        curr_adj = torch.FloatTensor(curr_adj)

        ref_feat = torch.FloatTensor(ref_feat)
        curr_feat = torch.FloatTensor(curr_feat) 

        #P = torch.randn(ref_adj.shape[0], curr_adj.shape[0], requires_grad=True)
        P = nn.Parameter(torch.FloatTensor(ref_adj.shape[0], curr_adj.shape[0]))
        with torch.no_grad():
            nn.init.constant_(P, 1.0/ref_adj.shape[0])
            P[ref_node_idx, :] = 0.0
            P[:, curr_node_idx] = 0.0
            P[ref_node_idx, curr_node_idx] = 1.0
        opt = torch.optim.Adam([P], lr=.01, betas=(0.5, 0.999))
        for i in range(args.align_steps):
            opt.zero_grad()
            feat_loss  = torch.norm(P @ curr_feat - ref_feat)

            aligned_adj = P @ curr_adj @ torch.transpose(P, 0, 1)
            align_loss = torch.norm(aligned_adj - ref_adj)
            loss =  feat_loss + align_loss
            loss.backward() # Calculate gradients
            self.writer.add_scalar('optimization/align_loss', loss, i)
            print('iter: ', i, '; loss: ', loss)
            opt.step()

        return P, aligned_adj, P @ curr_feat

    def explain_nodes(self, node_indices, args, graph_idx=0):
        masked_adjs = [self.explain(node_idx, graph_idx=graph_idx) for node_idx in node_indices]

        ref_idx = node_indices[0]
        ref_adj = masked_adjs[0]
        curr_idx = node_indices[1]
        curr_adj = masked_adjs[1]
        new_ref_idx, _, ref_feat,_,_ = self.extract_neighborhood(ref_idx)
        new_curr_idx, _, curr_feat,_,_   = self.extract_neighborhood(curr_idx)

        G_ref = io_utils.denoise_graph(ref_adj, new_ref_idx, ref_feat, threshold=0.1)
        denoised_ref_feat = np.array([G_ref.node[node]['feat'] for node in G_ref.nodes()])
        denoised_ref_adj = nx.to_numpy_matrix(G_ref)
        # ref center node
        ref_node_idx = list(G_ref.nodes()).index(new_ref_idx)

        G_curr = io_utils.denoise_graph(curr_adj, new_curr_idx, curr_feat, threshold=0.1)
        denoised_curr_feat = np.array([G_curr.node[node]['feat'] for node in G_curr.nodes()])
        denoised_curr_adj = nx.to_numpy_matrix(G_curr)
        # curr center node
        curr_node_idx = list(G_curr.nodes()).index(new_curr_idx)

        P, aligned_adj, aligned_feat = self.align(denoised_ref_feat, denoised_ref_adj, ref_node_idx,
                denoised_curr_feat, denoised_curr_adj, curr_node_idx, args=args)
        io_utils.log_matrix(self.writer, P, 'align/P', 0)

        G_ref = nx.convert_node_labels_to_integers(G_ref)
        io_utils.log_graph(self.writer, G_ref, 'align/ref')
        G_curr = nx.convert_node_labels_to_integers(G_curr)
        io_utils.log_graph(self.writer, G_curr, 'align/before')

        P = P.cpu().detach().numpy()
        aligned_adj = aligned_adj.cpu().detach().numpy()
        aligned_feat = aligned_feat.cpu().detach().numpy()

        aligned_idx = np.argmax(P[:, curr_node_idx])
        #print(list(G_curr.nodes()))
        print('aligned self: ', aligned_idx)
        #print('feat: ', aligned_feat.shape)
        #print('aligned adj: ', aligned_adj.shape)
        G_aligned = io_utils.denoise_graph(aligned_adj, aligned_idx, aligned_feat, threshold=0.5)
        io_utils.log_graph(self.writer, G_aligned, 'mask/aligned')
        
        #io_utils.log_graph(self.writer, aligned_adj.cpu().detach().numpy(), new_curr_idx,
        #        'align/aligned', epoch=1)

        return masked_adjs

    def log_representer(self, rep_val, sim_val, alpha, graph_idx=0):

        rep_val = rep_val.cpu().detach().numpy()
        sim_val = sim_val.cpu().detach().numpy()
        alpha = alpha.cpu().detach().numpy()
        sorted_rep = sorted(range(len(rep_val)), key=lambda k: rep_val[k])
        print(sorted_rep)
        topk = 5
        most_neg_idx = [sorted_rep[i] for i in range(topk)]
        most_pos_idx = [sorted_rep[-i-1] for i in range(topk)]
        rep_idx = [most_pos_idx, most_neg_idx]

        pred = np.argmax(self.pred[graph_idx][self.train_idx], axis=1)
        print(metrics.confusion_matrix(self.label[graph_idx][self.train_idx], pred))
        plt.switch_backend('agg')
        fig = plt.figure(figsize=(5,3), dpi=600)
        for i in range(2):
            for j in range(topk):
                idx = self.train_idx[rep_idx[i][j]]
                print('node idx: ', idx,
                        '; node label: ', self.label[graph_idx][idx],
                        '; pred: ', np.argmax(self.pred[graph_idx][idx])
                        )

                idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(idx,
                        graph_idx)
                G = nx.from_numpy_matrix(sub_adj)
                node_colors = [1 for i in range(G.number_of_nodes())]
                node_colors[idx_new] = 0
                #node_color='#336699',

                ax = plt.subplot(2, topk, i*topk+j+1)
                nx.draw(G, pos=nx.spring_layout(G), with_labels=True, font_size=4,
                        node_color=node_colors, cmap=plt.get_cmap('Set1'), vmin=0, vmax=8,
                        edge_vmin=0.0, edge_vmax=1.0,
                        width=0.5, node_size=25,
                        alpha=0.7)
                ax.xaxis.set_visible(False)
        fig.canvas.draw()
        self.writer.add_image('local/representer_neigh', tensorboardX.utils.figure_to_image(fig), 0)


        #fig = plt.figure(figsize=(4,3), dpi=400)
        #dat = [[i, rep_val[i], sim_val[i], alpha[i]] for i in range(len(rep_val))]
        #dat = pd.DataFrame(dat, columns=['idx', 'rep val', 'sim_val', 'alpha'])
        #sns.barplot(x='idx', y='rep val', data=dat)
        #fig.axes[0].xaxis.set_visible(False)
        #fig.canvas.draw()
        #self.writer.add_image('local/representer_bar', tensorboardX.utils.figure_to_image(fig), 0)

        #fig = plt.figure(figsize=(4,3), dpi=400)
        #sns.barplot(x='idx', y='alpha', data=dat)
        #fig.axes[0].xaxis.set_visible(False)
        #fig.canvas.draw()
        #self.writer.add_image('local/alpha_bar', tensorboardX.utils.figure_to_image(fig), 0)

        #fig = plt.figure(figsize=(4,3), dpi=400)
        #sns.barplot(x='idx', y='sim_val', data=dat)
        #fig.axes[0].xaxis.set_visible(False)
        #fig.canvas.draw()
        #self.writer.add_image('local/sim_bar', tensorboardX.utils.figure_to_image(fig), 0)


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
        self.mask_act = args.mask_act
        self.use_sigmoid = use_sigmoid

        init_strategy='normal'
        num_nodes = adj.size()[1]
        self.mask, self.mask_bias = self.construct_edge_mask(num_nodes, init_strategy=init_strategy)
        self.feat_mask = self.construct_feat_mask(x.size(-1), init_strategy='constant')
        params = [self.mask, self.feat_mask]
        if self.mask_bias is not None:
            params.append(self.mask_bias)
        # For masking diagonal entries
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        if args.gpu:
            self.diag_mask = self.diag_mask.cuda()

        self.scheduler, self.optimizer = train_utils.build_optimizer(args, params)

        self.coeffs = {'size': 0.0005, 'feat_size': 1.0, 'ent': 1.0,
                'feat_ent':0.1, 'grad': 0, 'lap': 1.0}

        # ypred = self.model(x, adj)[graph_idx][node_idx]
        # print('rerun pred: ', ypred)
        # masked_adj = adj * self.mask
        # ypred = self.model(x, masked_adj)
        # print('init mask pred: ', ypred[graph_idx][node_idx])

    def construct_feat_mask(self, feat_dim, init_strategy='normal'):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == 'normal':
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == 'constant':
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
                #mask[0] = 2
        return mask

    def construct_edge_mask(self, num_nodes, init_strategy='normal', const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == 'normal':
            std = nn.init.calculate_gain('relu') * math.sqrt(2.0 / (num_nodes + num_nodes))
            with torch.no_grad():
                mask.normal_(1.0, std)
                #mask.clamp_(0.0, 1.0)
        elif init_strategy == 'const':
            nn.init.constant_(mask, const_val)

        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None
        #print(mask)
        return mask, mask_bias

    def _masked_adj(self):
        sym_mask = self.mask
        if self.mask_act == 'sigmoid':
            sym_mask = torch.sigmoid(self.mask)
        elif self.mask_act == 'ReLU':
            sym_mask = nn.ReLU()(self.mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.adj.cuda() if self.args.gpu else self.adj
        masked_adj = adj * sym_mask
        if self.args.mask_bias:
            bias = (self.mask_bias + self.mask_bias.t()) / 2
            bias = nn.ReLU6()(bias * 6) / 6
            masked_adj += (bias + bias.t()) / 2
        return masked_adj * self.diag_mask

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum

    def forward(self, node_idx, unconstrained=False):
        x = self.x.cuda() if self.args.gpu else self.x

        if unconstrained:
            sym_mask = torch.sigmoid(self.mask) if self.use_sigmoid else self.mask
            self.masked_adj = torch.unsqueeze((sym_mask + sym_mask.t()) / 2, 0) * self.diag_mask
        else:
            self.masked_adj = self._masked_adj()
            feat_mask = torch.sigmoid(self.feat_mask) if self.use_sigmoid else self.feat_mask
            #std_tensor = torch.ones_like(x, dtype=torch.float) / 2
            #mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
            #z = torch.normal(mean=mean_tensor, std=std_tensor)
            x = x * feat_mask
            #x = x + z * (1 - feat_mask)

        ypred = self.model(x, self.masked_adj)
        node_pred = ypred[self.graph_idx, node_idx, :]
        return nn.Softmax(dim=0)(node_pred)

    def adj_feat_grad(self, node_idx, pred_label_node):
        self.model.zero_grad()
        self.adj.requires_grad = True
        self.x.requires_grad = True
        if self.adj.grad is not None:
            self.adj.grad.zero_()
            self.x.grad.zero_()
        if self.args.gpu:
            adj = self.adj.cuda()
            x = self.x.cuda()
            label = self.label.cuda()
        else:
            x, adj = self.x, self.adj
        ypred = self.model(x, adj)
        logit = nn.Softmax(dim=0)(ypred[self.graph_idx, node_idx, :])
        logit = logit[pred_label_node]
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
        gt_label_node = self.label[0][node_idx]
        logit = pred[gt_label_node]
        pred_loss = -torch.log(logit)

        # size
        mask = self.mask
        if self.mask_act == 'sigmoid':
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == 'ReLU':
            mask = nn.ReLU()(self.mask)
        size_loss = self.coeffs['size'] * torch.sum(mask)

        #pre_mask_sum = torch.sum(self.feat_mask)
        feat_mask = torch.sigmoid(self.feat_mask) if self.use_sigmoid else self.feat_mask
        feat_size_loss = self.coeffs['feat_size'] * torch.mean(feat_mask) 

        # entropy
        mask_ent = -mask * torch.log(mask) - (1-mask) * torch.log(1-mask)
        mask_ent_loss = self.coeffs['ent'] * torch.mean(mask_ent)

        feat_mask_ent = -feat_mask * torch.log(feat_mask) - (1-feat_mask) * torch.log(1-feat_mask)
        feat_mask_ent_loss = self.coeffs['feat_ent'] * torch.mean(feat_mask_ent)


        # laplacian
        D = torch.diag(torch.sum(self.masked_adj[0], 0))
        L = D - self.masked_adj[self.graph_idx]
        pred_label_t = torch.tensor(pred_label, dtype=torch.float)
        if self.args.gpu:
            pred_label_t = pred_label_t.cuda()
            L = L.cuda()
        lap_loss = self.coeffs['lap'] * (pred_label_t @ L @ pred_label_t) / self.adj.numel()

        # grad
        # adj
        #adj_grad, x_grad = self.adj_feat_grad(node_idx, pred_label_node)
        #adj_grad = adj_grad[self.graph_idx]
        #x_grad = x_grad[self.graph_idx]
        #if self.args.gpu:
        #    adj_grad = adj_grad.cuda()
        #grad_loss = self.coeffs['grad'] * -torch.mean(torch.abs(adj_grad) * mask)

        # feat
        # x_grad_sum = torch.sum(x_grad, 1)
        #grad_feat_loss = self.coeffs['featgrad'] * -torch.mean(x_grad_sum * mask)

        loss = pred_loss + size_loss + lap_loss + mask_ent_loss + feat_size_loss
        if self.writer is not None:
            self.writer.add_scalar('optimization/size_loss', size_loss, epoch)
            self.writer.add_scalar('optimization/feat_size_loss', feat_size_loss, epoch)
            self.writer.add_scalar('optimization/mask_ent_loss', mask_ent_loss, epoch)
            self.writer.add_scalar('optimization/feat_mask_ent_loss', mask_ent_loss, epoch)
            #self.writer.add_scalar('optimization/grad_loss', grad_loss, epoch)
            self.writer.add_scalar('optimization/pred_loss', pred_loss, epoch)
            self.writer.add_scalar('optimization/lap_loss', lap_loss, epoch)
            self.writer.add_scalar('optimization/overall_loss', loss, epoch)
        return loss

    def log_mask(self, epoch):
        plt.switch_backend('agg')
        fig = plt.figure(figsize=(4,3), dpi=400)
        plt.imshow(self.mask.cpu().detach().numpy(), cmap=plt.get_cmap('BuPu'))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.writer.add_image('mask/mask', tensorboardX.utils.figure_to_image(fig), epoch)

        fig = plt.figure(figsize=(4,3), dpi=400)
        plt.imshow(self.feat_mask.cpu().detach().numpy()[:,np.newaxis], cmap=plt.get_cmap('BuPu'))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.writer.add_image('mask/feat_mask', tensorboardX.utils.figure_to_image(fig), epoch)

        fig = plt.figure(figsize=(4,3), dpi=400)
        # use [0] to remove the batch dim
        plt.imshow(self.masked_adj[0].cpu().detach().numpy(), cmap=plt.get_cmap('BuPu'))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.writer.add_image('mask/adj', tensorboardX.utils.figure_to_image(fig), epoch)

        if self.args.mask_bias:
            fig = plt.figure(figsize=(4,3), dpi=400)
            # use [0] to remove the batch dim
            plt.imshow(self.mask_bias.cpu().detach().numpy(), cmap=plt.get_cmap('BuPu'))
            cbar = plt.colorbar()
            cbar.solids.set_edgecolor("face")

            plt.tight_layout()
            fig.canvas.draw()
            self.writer.add_image('mask/bias', tensorboardX.utils.figure_to_image(fig), epoch)


    def log_adj_grad(self, node_idx, pred_label, epoch):
        if self.adj.grad is not None:
            io_utils.log_matrix(self.writer, self.adj.grad.squeeze(), 'grad/adj1', epoch)
        adj_grad = torch.abs(self.adj_feat_grad(node_idx, pred_label[node_idx])[0])[self.graph_idx]
        adj_grad = adj_grad + adj_grad.t()
        io_utils.log_matrix(self.writer, adj_grad, 'grad/adj', epoch)
        #self.adj.requires_grad = False

    def log_masked_adj(self, node_idx, epoch, label=None):
        # use [0] to remove the batch dim
        masked_adj = self.masked_adj[0].cpu().detach().numpy()
        G = io_utils.denoise_graph(masked_adj, node_idx, label=label)
        io_utils.log_graph(self.writer, G, name='mask/graph', epoch=epoch)
        #num_nodes = self.adj.size()[-1]
        #G = nx.Graph()
        #G.add_nodes_from(range(num_nodes))
        #G.node[node_idx]['color'] = 0
        #weighted_edge_list = [(i, j, masked_adj[i, j]) for i in range(num_nodes) for j in range(num_nodes) if masked_adj[i,j] > 0.1]
        #G.add_weighted_edges_from(weighted_edge_list)
        #Gc = max(nx.connected_component_subgraphs(G), key=len)
        #edge_colors = [Gc[i][j]['weight'] for (i,j) in Gc.edges()]
        #node_colors = [Gc.node[i]['color'] if 'color' in Gc.node[i] else 1 for i in Gc.nodes()]

        #plt.switch_backend('agg')
        #fig = plt.figure(figsize=(4,3), dpi=600)
        #nx.draw(Gc, pos=nx.spring_layout(G), with_labels=True, font_size=4,
        #        node_color=node_colors, vmin=0, vmax=8, cmap=plt.get_cmap('Set1'),
        #        edge_color=edge_colors, edge_cmap=plt.get_cmap('Greys'), edge_vmin=0.0, edge_vmax=1.0,
        #        width=0.5, node_size=25,
        #        alpha=0.7)
        #fig.axes[0].xaxis.set_visible(False)
        #fig.canvas.draw()
        #self.writer.add_image('mask/graph', tensorboardX.utils.figure_to_image(fig), epoch)

