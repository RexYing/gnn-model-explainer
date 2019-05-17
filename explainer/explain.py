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
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_auc_score
import pdb
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve



# import models_gcn.GCN as GCN
from models_gcn import GCN

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
            print_training=True, graph_mode=False, graph_idx=False):
        self.model = model
        self.model.eval()
        self.adj = adj
        self.feat = feat
        self.label = label
        self.pred = pred
        self.train_idx = train_idx
        self.n_hops = args.num_gc_layers
        self.graph_mode = graph_mode
        self.graph_idx = graph_idx
        self.neighborhoods = None if self.graph_mode else  self._neighborhoods()
        self.args = args
        self.writer = writer
        self.print_training = print_training
        #self.representer()

   

    def _neighborhoods(self):
        adj = torch.tensor(self.adj, dtype=torch.float).cuda()
        hop_adj = power_adj = adj
        for i in range(self.n_hops-1):
            power_adj = power_adj @ adj
            hop_adj = hop_adj + power_adj
            hop_adj = (hop_adj > 0).float()
        return hop_adj.cpu().numpy()

    def representer(self):
        self.model.train()
        self.model.zero_grad()
        adj = torch.tensor(self.adj, dtype=torch.float)
        x = torch.tensor(self.feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(self.label, dtype=torch.long)
        if self.args.gpu:
            adj, x, label = adj.cuda(), x.cuda(), label.cuda()

        preds,_ = self.model(x, adj)
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

    def explain(self, node_idx, graph_idx=0, graph_mode=False, unconstrained=False, model='exp'):
        '''Explain a single node prediction
        '''
        # index of the query node in the new adj
        if graph_mode:
          node_idx_new = node_idx
          sub_adj = self.adj[graph_idx]
          sub_feat = self.feat[graph_idx,:]
          sub_label = self.label[graph_idx]
          neighbors = np.asarray(range(self.adj.shape[0]))
        else:
          print('node label: ', self.label[graph_idx][node_idx])
          node_idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(node_idx, graph_idx)
          print('neigh graph idx: ', node_idx, node_idx_new)
          sub_label = np.expand_dims(sub_label, axis=0)
        sub_adj = np.expand_dims(sub_adj, axis=0)
        sub_feat = np.expand_dims(sub_feat, axis=0)

        adj = torch.tensor(sub_adj, dtype=torch.float)
        x = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(sub_label, dtype=torch.long)
        #print('loaded pred: ', self.pred[graph_idx][node_idx])
        if self.graph_mode:
          pred_label = np.argmax(self.pred[0][graph_idx], axis=0)
          print('pred label: ', pred_label)
        else:
          pred_label = np.argmax(self.pred[graph_idx][neighbors], axis=1)
          print('pred label: ', pred_label[node_idx_new])

        #f_test = self.embedding[graph_idx, node_idx, :]
        #f_idx = self.embedding[graph_idx, self.train_idx, :]
        #alpha = self.alpha[graph_idx, self.train_idx, pred_label[node_idx_new]]
        #sim_val = f_idx @ f_test
        #rep_val = sim_val * alpha
        #if self.writer is not None:
        #    self.log_representer(rep_val, sim_val, alpha)

        explainer = ExplainModule(adj, x, self.model, label, self.args, writer=self.writer, graph_idx=self.graph_idx, graph_mode=self.graph_mode)
        if self.args.gpu:
            explainer = explainer.cuda()

        self.model.eval()

        # gradient baseline
        if model=='grad':
            explainer.zero_grad()
            # pdb.set_trace()
            adj_grad = torch.abs(explainer.adj_feat_grad(node_idx_new, pred_label[node_idx_new])[0])[graph_idx]
            masked_adj = (adj_grad + adj_grad.t())
            masked_adj = nn.functional.sigmoid(masked_adj)
            masked_adj = masked_adj.cpu().detach().numpy()*sub_adj.squeeze()

        else:
            explainer.train()
            begin_time = time.time()
            for epoch in range(self.args.num_epochs):
                explainer.zero_grad()
                explainer.optimizer.zero_grad()
                ypred,adj_att = explainer(node_idx_new, unconstrained=unconstrained)
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

                single_subgraph_label = sub_label.squeeze()

                if self.writer is not None:
                    self.writer.add_scalar('mask/density', mask_density, epoch)
                    self.writer.add_scalar('optimization/lr', explainer.optimizer.param_groups[0]['lr'], epoch)
                    if epoch % 50 == 0:
                        explainer.log_mask(epoch)
                        explainer.log_masked_adj(node_idx_new, epoch, label=single_subgraph_label)
                        explainer.log_adj_grad(node_idx_new, pred_label, epoch,
                                label=single_subgraph_label)

                if model != 'exp':
                    break

            print('finished training in ', time.time() - begin_time)
            if model =='exp':
                masked_adj = explainer.masked_adj[0].cpu().detach().numpy()*sub_adj.squeeze()
            else:
                adj_att = nn.functional.sigmoid(adj_att).squeeze()
                masked_adj = adj_att.cpu().detach().numpy()*sub_adj.squeeze()

        return masked_adj

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


    # def make_pred_real(self,G,start):
    #     # house graph
    #     if self.args.dataset == 'syn1' or self.args.dataset == 'syn2':
    #         num_pred = max(G.number_of_edges(),6)
    #         real = np.ones(num_pred)
    #         pred = np.zeros(num_pred)
    #         if G.has_edge(start, start+1):
    #             pred[0] = G[start][start+1]['weight']
    #         if G.has_edge(start+1, start+2):
    #             pred[1] = 1
    #         if G.has_edge(start+2, start+3):
    #             pred[2] = 1
    #         if G.has_edge(start+3, start):
    #             pred[3] = 1
    #         if G.has_edge(start+4, start):
    #             pred[4] = 1
    #         if G.has_edge(start+4, start+1):
    #             pred[5] = 1
    #         pdb.set_trace()
    #         precision = precision_score(real,pred)
    #         recall = recall_score(real,pred)
    #
    #     # cycle graph
    #     elif self.args.dataset == 'syn4':
    #         num_pred = max(G.number_of_edges(),6)
    #         real = np.ones(num_pred)
    #         pred = np.zeros(num_pred)
    #         if G.has_edge(start, start+1):
    #             pred[0] = 1
    #         if G.has_edge(start+1, start+2):
    #             pred[1] = 1
    #         if G.has_edge(start+2, start+3):
    #             pred[2] = 1
    #         if G.has_edge(start+3, start+4):
    #             pred[3] = 1
    #         if G.has_edge(start+4, start+5):
    #             pred[4] = 1
    #         if G.has_edge(start+5, start):
    #             pred[5] = 1
    #
    #         precision = precision_score(real,pred)
    #         recall = recall_score(real,pred)
    #
    #     else:
    #         precision, recall = 0, 0
    #
    #     return precision,recall



    def make_pred_real(self, adj, start):
        # house graph
        if self.args.dataset == 'syn1' or self.args.dataset == 'syn2':
            # num_pred = max(G.number_of_edges(), 6)
            pred = adj[np.triu(adj) > 0]
            real = adj.copy()

            if real[start][start + 1]>0:
                real[start][start + 1] = 10
            if real[start + 1][start + 2]>0:
                real[start + 1][start + 2] = 10
            if real[start + 2][start + 3]>0:
                real[start + 2][start + 3] = 10
            if real[start][start + 3]>0:
                real[start][start + 3] = 10
            if real[start][start + 4]>0:
                real[start][start + 4] = 10
            if real[start + 1][start + 4]:
                real[start + 1][start + 4] = 10
            real = real[np.triu(real) > 0]
            real[real!=10] = 0
            real[real==10] = 1

            # auc = roc_auc_score(real, pred)

        # cycle graph
        elif self.args.dataset == 'syn4':
            pred = adj[np.triu(adj) > 0]
            real = adj.copy()
            # pdb.set_trace()
            if real[start][start + 1] > 0:
                real[start][start + 1] = 10
            if real[start + 1][start + 2] > 0:
                real[start + 1][start + 2] = 10
            if real[start + 2][start + 3] > 0:
                real[start + 2][start + 3] = 10
            if real[start+3][start + 4] > 0:
                real[start+3][start + 4] = 10
            if real[start+4][start + 5] > 0:
                real[start+4][start + 5] = 10
            if real[start][start + 5]:
                real[start][start + 5] = 10
            real = real[np.triu(real) > 0]
            real[real != 10] = 0
            real[real == 10] = 1

        return pred, real


    def explain_nodes_gnn_stats(self, node_indices, args, graph_idx=0,model='grad'):
        masked_adjs = [self.explain(node_idx, graph_idx=graph_idx, model=model) for node_idx in node_indices]
        # pdb.set_trace()
        graphs = []
        feats = []
        adjs = []
        pred_all = []
        real_all = []
        for i,idx in enumerate(node_indices):
            new_idx, _, feat, _, _ = self.extract_neighborhood(idx)
            G = io_utils.denoise_graph(masked_adjs[i], new_idx, feat,threshold_num=20)
            pred,real = self.make_pred_real(masked_adjs[i],new_idx)
            pred_all.append(pred)
            real_all.append(real)
            denoised_feat = np.array([G.node[node]['feat'] for node in G.nodes()])
            denoised_adj = nx.to_numpy_matrix(G)
            graphs.append(G)
            feats.append(denoised_feat)
            adjs.append(denoised_adj)
            io_utils.log_graph(self.writer, G,
                               'graph/{}_{}_{}'.format(self.args.dataset,model,i), identify_self=True)

        pred_all = np.concatenate((pred_all),axis=0)
        real_all = np.concatenate((real_all),axis=0)

        auc_all = roc_auc_score(real_all,pred_all)
        precision, recall, thresholds = precision_recall_curve(real_all,pred_all)

        plt.switch_backend('agg')
        plt.plot(recall, precision)
        plt.savefig('log/pr/pr_'+self.args.dataset+'_'+model+'.png')

        plt.close()

        # plot PR
        pred_all = np.concatenate((pred_all),axis=0)
        real_all = np.concatenate((real_all),axis=0)

        auc_all = roc_auc_score(real_all,pred_all)
        precision, recall, thresholds = precision_recall_curve(real_all,pred_all)

        plt.switch_backend('agg')
        plt.plot(recall, precision)
        plt.savefig('log/pr/pr_'+self.args.dataset+'_'+model+'.png')

        plt.close()

        with open('log/pr/auc_'+self.args.dataset+'_'+model+'.txt', 'w') as f:
            f.write('dataset: {}, model: {}, auc: {}\n'.format(self.args.dataset, 'exp', str(auc_all)))

        return masked_adjs


    def explain_nodes_gnn_cluster(self, node_indices, args, graph_idx=0):
        masked_adjs = [self.explain(node_idx, graph_idx=graph_idx) for node_idx in node_indices]

        # ref_idx = node_indices[0]
        # ref_adj = masked_adjs[0]

        graphs = []
        feats = []
        adjs = []
        for i,idx in enumerate(node_indices):
            new_idx, _, feat, _, _ = self.extract_neighborhood(idx)
            G = io_utils.denoise_graph(masked_adjs[i], new_idx, feat, threshold=0.1)
            denoised_feat = np.array([G.node[node]['feat'] for node in G.nodes()])
            denoised_adj = nx.to_numpy_matrix(G)
            graphs.append(G)
            feats.append(denoised_feat)
            adjs.append(denoised_adj)

        model = GCN(input_dim=feats[0].shape[1],
                    hidden_dim=64, output_dim=64, num_layers=2,
                    normalize_embedding_l2=True).cuda()
        # model = self.model
        pred_all = []

        for i in range(len(graphs)):
            adj = Variable(torch.from_numpy(adjs[i]).float(), requires_grad=False).cuda()
            feature = Variable(torch.from_numpy(feats[i]).float(), requires_grad=False).cuda()
            # import pdb
            # pdb.set_trace()
            pred = model(feature, adj)
            pred_all.append(pred.data.cpu().numpy())

        X = np.concatenate(pred_all, axis=0)

        # pdb.set_trace()
        from sklearn.manifold import TSNE

        plt.switch_backend('agg')
        # X = TSNE(n_components=2, n_iter=1000).fit_transform(X)
        # plt.scatter(X[:, 0], X[:, 1])

        # db = DBSCAN(eps=0.2, min_samples=3).fit(X)
        db = DBSCAN(eps=0.2, min_samples=len(graphs)//4).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)


        count_dict = np.bincount(db.labels_[db.labels_>=0])
        graphs_denoised = []
        counter = 0
        for i in range(len(graphs)):
            graph = graphs[i]
            nodes = []
            for node in graph.nodes():
                label = db.labels_[counter]
                if label>=0 and count_dict[label] < len(graphs)*2:
                    nodes.append(node)
                counter += 1
            graphs_denoised.append(graph.subgraph(nodes))


        for i,graph in enumerate(graphs):
            io_utils.log_graph(self.writer, graph, 'align/before'+str(i))
        for i,graph in enumerate(graphs_denoised):
            io_utils.log_graph(self.writer, graph, 'align/after'+str(i))

        #
        #
        # # Black removed and is used for noise instead.
        # unique_labels = set(labels)
        # colors = [plt.cm.Spectral(each)
        #           for each in np.linspace(0, 1, len(unique_labels))]
        # for k, col in zip(unique_labels, colors):
        #     if k == -1:
        #         # Black used for noise.
        #         col = [0, 0, 0, 1]
        #
        #     class_member_mask = (labels == k)
        #
        #     xy = X[class_member_mask & core_samples_mask]
        #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #              markeredgecolor='k', markersize=14)
        #
        #     xy = X[class_member_mask & ~core_samples_mask]
        #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #              markeredgecolor='k', markersize=6)
        #
        # plt.title('Estimated number of clusters: %d' % n_clusters_)
        #
        # plt.savefig('log/fig/test.png')

        # io_utils.log_graph(self.writer, graphs[0], 'align/ref')
        # io_utils.log_graph(self.writer, graphs[1], 'align/before')
        #io_utils.log_graph(self.writer, aligned_adj.cpu().detach().numpy(), new_curr_idx,
        #        'align/aligned', epoch=1)

        return masked_adjs

    def explain_graphs(self, graph_indices):
        masked_adjs = []

        for graph_idx in graph_indices:
          masked_adj = self.explain(node_idx=0, graph_idx=graph_idx, graph_mode=True)
          G_denoised = io_utils.denoise_graph(masked_adj, 0, threshold_num=20,
                  feat=self.feat[graph_idx], max_component=False)
          label = self.label[graph_idx]
          io_utils.log_graph(self.writer, G_denoised, 
                  'graph/graphidx_{}_label={}'.format(graph_idx, label), identify_self=False, nodecolor='feat')
          masked_adjs.append(masked_adj)
          
          G_orig = io_utils.denoise_graph(self.adj[graph_idx], 0, feat=self.feat[graph_idx],
                  threshold=None, max_component=False) 
          #G_orig = nx.from_numpy_matrix(self.adj[graph_idx].cpu().detach().numpy())
          io_utils.log_graph(self.writer, G_orig, 'graph/graphidx_{}'.format(graph_idx), identify_self=False, nodecolor='feat')

        # plot cmap for graphs' node features
        io_utils.plot_cmap_tb(self.writer, 'tab20', 20, 'tab20_cmap')

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

        if self.graph_mode:
          pred = np.argmax(self.pred[0][graph_idx], axis=0)
        else:
          pred = np.argmax(self.pred[graph_idx][self.train_idx], axis=1)
        print(metrics.confusion_matrix(self.label[graph_idx][self.train_idx], pred))
        plt.switch_backend('agg')
        fig = plt.figure(figsize=(5,3), dpi=600)
        for i in range(2):
            for j in range(topk):
                idx = self.train_idx[rep_idx[i][j]]
                print('node idx: ', idx,
                        '; node label: ', self.label[graph_idx][idx],
                        '; pred: ', pred
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
    def __init__(self, adj, x, model, label, args, graph_idx=0, writer=None, use_sigmoid=True, graph_mode=False):
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
        self.graph_mode = graph_mode

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

        self.coeffs = {'size': 0.005, 'feat_size': 1.0, 'ent': 1.0,
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
            marginalize = False
            if marginalize:
                std_tensor = torch.ones_like(x, dtype=torch.float) / 2
                mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
                z = torch.normal(mean=mean_tensor, std=std_tensor)
                x = x + z * (1 - feat_mask)
            else:
                x = x * feat_mask

        ypred,adj_att = self.model(x, self.masked_adj)
        if self.graph_mode:
          res = nn.Softmax(dim=0)(ypred[0]) 
        else:
          node_pred = ypred[self.graph_idx, node_idx, :]
          res = nn.Softmax(dim=0)(node_pred)
        # return res
        return res,adj_att

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
        ypred,_ = self.model(x, adj)
        if self.graph_mode:
          logit = nn.Softmax(dim=0)(ypred[0])
        else:
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
        mi_obj = False
        if mi_obj:
            pred_loss = - torch.sum(pred * torch.log(pred))
        else:
            pred_label_node = pred_label if self.graph_mode else pred_label[node_idx]
            gt_label_node = self.label if self.graph_mode else self.label[0][node_idx]
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
        m_adj = self.masked_adj if self.graph_mode else self.masked_adj[self.graph_idx]
        L = D - m_adj 
        pred_label_t = torch.tensor(pred_label, dtype=torch.float)
        if self.args.gpu:
            pred_label_t = pred_label_t.cuda()
            L = L.cuda()
        if self.graph_mode:
          lap_loss = 0
        else:
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

        #fig = plt.figure(figsize=(4,3), dpi=400)
        #plt.imshow(self.feat_mask.cpu().detach().numpy()[:,np.newaxis], cmap=plt.get_cmap('BuPu'))
        #cbar = plt.colorbar()
        #cbar.solids.set_edgecolor("face")

        #plt.tight_layout()
        #fig.canvas.draw()
        #self.writer.add_image('mask/feat_mask', tensorboardX.utils.figure_to_image(fig), epoch)
        io_utils.log_matrix(self.writer, torch.sigmoid(self.feat_mask), 'mask/feat_mask', epoch)
        #print(self.feat_mask,  'FEAT MASK ----------------')

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


    def log_adj_grad(self, node_idx, pred_label, epoch, label=None):
        if self.graph_mode:
          predicted_label = pred_label
          #adj_grad, x_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])[0]
          adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label)
          adj_grad = torch.abs(adj_grad)[0]
          x_grad = torch.sum(x_grad[0], 0, keepdim=True).t()
        else:
          predicted_label = pred_label[node_idx]
          #adj_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])[self.graph_idx]
          adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label)
          adj_grad = torch.abs(adj_grad)[self.graph_idx]
          x_grad = x_grad[self.graph_idx][node_idx][:, np.newaxis]
          #x_grad = torch.sum(x_grad[self.graph_idx], 0, keepdim=True).t()
        adj_grad = (adj_grad + adj_grad.t()) / 2
        io_utils.log_matrix(self.writer, adj_grad, 'grad/adj', epoch)
        adj_grad = (adj_grad * self.adj).squeeze()
        io_utils.log_matrix(self.writer, adj_grad, 'grad/adj1', epoch)
        #self.adj.requires_grad = False
        io_utils.log_matrix(self.writer, self.adj.squeeze(), 'grad/adj_orig', epoch)

        masked_adj = self.masked_adj[0].cpu().detach().numpy()
        G = io_utils.denoise_graph(masked_adj, node_idx, feat=self.x[0], threshold=None,
                max_component=False)
        io_utils.log_graph(self.writer, G, name='grad/graph_orig', epoch=epoch, identify_self=False,
                label_node_feat=True, nodecolor='feat', edge_vmax=None, args=self.args)
        io_utils.log_matrix(self.writer, x_grad, 'grad/feat', epoch)
        #print(x_grad,  'X GRAD ----------------')

        adj_grad = adj_grad.detach().numpy()
        if self.graph_mode:
            print('GRAPH model')
            G = io_utils.denoise_graph(adj_grad, node_idx, feat=self.x[0], threshold_num=20,
                    max_component=False)
            io_utils.log_graph(self.writer, G, name='grad/graph', epoch=epoch, identify_self=False,
                    label_node_feat=True, nodecolor='feat', edge_vmax=None, args=self.args)
        else:
            #G = io_utils.denoise_graph(adj_grad, node_idx, label=label, threshold=0.5)
            G = io_utils.denoise_graph(adj_grad, node_idx, threshold=0.001)
            io_utils.log_graph(self.writer, G, name='grad/graph', epoch=epoch, edge_vmax=0.008,
                    args=self.args)

    def log_masked_adj(self, node_idx, epoch, name='mask/graph', label=None):
        # use [0] to remove the batch dim
        masked_adj = self.masked_adj[0].cpu().detach().numpy()
        if self.graph_mode:
            G = io_utils.denoise_graph(masked_adj, node_idx, feat=self.x[0], threshold_num=20,
                    max_component=False)
            io_utils.log_graph(self.writer, G, name=name, identify_self=False,
                    nodecolor='feat', epoch=epoch, label_node_feat=True, edge_vmax=None, args=self.args)
        else:
            #G = io_utils.denoise_graph(masked_adj, node_idx, label=label)
            G = io_utils.denoise_graph(masked_adj, node_idx)
            io_utils.log_graph(self.writer, G, name=name, identify_self=True,
                    nodecolor='label', epoch=epoch, edge_vmax=0.7, args=self.args)

