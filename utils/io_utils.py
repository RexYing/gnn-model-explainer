""" io_utils.py

    Utilities for reading and writing logs.
"""
import os
import statistics
import re
import csv

import numpy as np
import pandas as pd
import scipy as sc


import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import torch
import networkx as nx
import tensorboardX

import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable

# Only necessary to rebuild the Chemistry example
# from rdkit import Chem

import utils.featgen as featgen

use_cuda = torch.cuda.is_available()


def gen_prefix(args):
    '''Generate label prefix for a graph model.
    '''
    if args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
    name += "_" + args.method

    name += "_h" + str(args.hidden_dim) + "_o" + str(args.output_dim)
    if not args.bias:
        name += "_nobias"
    if len(args.name_suffix) > 0:
        name += "_" + args.name_suffix
    return name


def gen_explainer_prefix(args):
    '''Generate label prefix for a graph explainer model.
    '''
    name = gen_prefix(args) + "_explain"
    if len(args.explainer_suffix) > 0:
        name += "_" + args.explainer_suffix
    return name


def create_filename(save_dir, args, isbest=False, num_epochs=-1):
    """
    Args:
        args        :  the arguments parsed in the parser
        isbest      :  whether the saved model is the best-performing one
        num_epochs  :  epoch number of the model (when isbest=False)
    """
    filename = os.path.join(save_dir, gen_prefix(args))
    os.makedirs(filename, exist_ok=True)

    if isbest:
        filename = os.path.join(filename, "best")
    elif num_epochs > 0:
        filename = os.path.join(filename, str(num_epochs))

    return filename + ".pth.tar"


def save_checkpoint(model, optimizer, args, num_epochs=-1, isbest=False, cg_dict=None):
    """Save pytorch model checkpoint.

    Args:
        - model         : The PyTorch model to save.
        - optimizer     : The optimizer used to train the model.
        - args          : A dict of meta-data about the model.
        - num_epochs    : Number of training epochs.
        - isbest        : True if the model has the highest accuracy so far.
        - cg_dict       : A dictionary of the sampled computation graphs.
    """
    filename = create_filename(args.ckptdir, args, isbest, num_epochs=num_epochs)
    torch.save(
        {
            "epoch": num_epochs,
            "model_type": args.method,
            "optimizer": optimizer,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cg": cg_dict,
        },
        filename,
    )


def load_ckpt(args, isbest=False):
    '''Load a pre-trained pytorch model from checkpoint.
    '''
    print("loading model")
    filename = create_filename(args.ckptdir, args, isbest)
    print(filename)
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        ckpt = torch.load(filename)
    else:
        print("Checkpoint does not exist!")
        print("Checked path -- {}".format(filename))
        print("Make sure you have provided the correct path!")
        print("You may have forgotten to train a model for this dataset.")
        print()
        print("To train one of the paper's models, run the following")
        print(">> python train.py --dataset=DATASET_NAME")
        print()
        raise Exception("File not found.")
    return ckpt

def preprocess_cg(cg):
    """Pre-process computation graph."""
    if use_cuda:
        preprocessed_cg_tensor = torch.from_numpy(cg).cuda()
    else:
        preprocessed_cg_tensor = torch.from_numpy(cg)

    preprocessed_cg_tensor.unsqueeze_(0)
    return Variable(preprocessed_cg_tensor, requires_grad=False)

def load_model(path):
    """Load a pytorch model."""
    model = torch.load(path)
    model.eval()
    if use_cuda:
        model.cuda()

    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = False

    return model


def load_cg(path):
    """Load a computation graph."""
    cg = pickle.load(open(path))
    return cg


def save(mask_cg):
    """Save a rendering of the computation graph mask."""
    mask = mask_cg.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))

    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask

    cv2.imwrite("mask.png", np.uint8(255 * mask))

def log_matrix(writer, mat, name, epoch, fig_size=(8, 6), dpi=200):
    """Save an image of a matrix to disk.

    Args:
        - writer    :  A file writer.
        - mat       :  The matrix to write.
        - name      :  Name of the file to save.
        - epoch     :  Epoch number.
        - fig_size  :  Size to of the figure to save.
        - dpi       :  Resolution.
    """
    plt.switch_backend("agg")
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    mat = mat.cpu().detach().numpy()
    if mat.ndim == 1:
        mat = mat[:, np.newaxis]
    plt.imshow(mat, cmap=plt.get_cmap("BuPu"))
    cbar = plt.colorbar()
    cbar.solids.set_edgecolor("face")

    plt.tight_layout()
    fig.canvas.draw()
    writer.add_image(name, tensorboardX.utils.figure_to_image(fig), epoch)


def denoise_graph(adj, node_idx, feat=None, label=None, threshold=None, threshold_num=None, max_component=True):
    """Cleaning a graph by thresholding its node values.

    Args:
        - adj               :  Adjacency matrix.
        - node_idx          :  Index of node to highlight (TODO ?)
        - feat              :  An array of node features.
        - label             :  A list of node labels.
        - threshold         :  The weight threshold.
        - theshold_num      :  The maximum number of nodes to threshold.
        - max_component     :  TODO
    """
    num_nodes = adj.shape[-1]
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.nodes[node_idx]["self"] = 1
    if feat is not None:
        for node in G.nodes():
            G.nodes[node]["feat"] = feat[node]
    if label is not None:
        for node in G.nodes():
            G.nodes[node]["label"] = label[node]

    if threshold_num is not None:
        # this is for symmetric graphs: edges are repeated twice in adj
        adj_threshold_num = threshold_num * 2
        #adj += np.random.rand(adj.shape[0], adj.shape[1]) * 1e-4
        neigh_size = len(adj[adj > 0])
        threshold_num = min(neigh_size, adj_threshold_num)
        threshold = np.sort(adj[adj > 0])[-threshold_num]

    if threshold is not None:
        weighted_edge_list = [
            (i, j, adj[i, j])
            for i in range(num_nodes)
            for j in range(num_nodes)
            if adj[i, j] >= threshold
        ]
    else:
        weighted_edge_list = [
            (i, j, adj[i, j])
            for i in range(num_nodes)
            for j in range(num_nodes)
            if adj[i, j] > 1e-6
        ]
    G.add_weighted_edges_from(weighted_edge_list)
    if max_component:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    else:
        # remove zero degree nodes
        G.remove_nodes_from(list(nx.isolates(G)))
    return G

# TODO: unify log_graph and log_graph2
def log_graph(
    writer,
    Gc,
    name,
    identify_self=True,
    nodecolor="label",
    epoch=0,
    fig_size=(4, 3),
    dpi=300,
    label_node_feat=False,
    edge_vmax=None,
    args=None,
):
    """
    Args:
        nodecolor: the color of node, can be determined by 'label', or 'feat'. For feat, it needs to
            be one-hot'
    """
    cmap = plt.get_cmap("Set1")
    plt.switch_backend("agg")
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    node_colors = []
    # edge_colors = [min(max(w, 0.0), 1.0) for (u,v,w) in Gc.edges.data('weight', default=1)]
    edge_colors = [w for (u, v, w) in Gc.edges.data("weight", default=1)]

    # maximum value for node color
    vmax = 8
    for i in Gc.nodes():
        if nodecolor == "feat" and "feat" in Gc.nodes[i]:
            num_classes = Gc.nodes[i]["feat"].size()[0]
            if num_classes >= 10:
                cmap = plt.get_cmap("tab20")
                vmax = 19
            elif num_classes >= 8:
                cmap = plt.get_cmap("tab10")
                vmax = 9
            break

    feat_labels = {}
    for i in Gc.nodes():
        if identify_self and "self" in Gc.nodes[i]:
            node_colors.append(0)
        elif nodecolor == "label" and "label" in Gc.nodes[i]:
            node_colors.append(Gc.nodes[i]["label"] + 1)
        elif nodecolor == "feat" and "feat" in Gc.nodes[i]:
            # print(Gc.nodes[i]['feat'])
            feat = Gc.nodes[i]["feat"].detach().numpy()
            # idx with pos val in 1D array
            feat_class = 0
            for j in range(len(feat)):
                if feat[j] == 1:
                    feat_class = j
                    break
            node_colors.append(feat_class)
            feat_labels[i] = feat_class
        else:
            node_colors.append(1)
    if not label_node_feat:
        feat_labels = None

    plt.switch_backend("agg")
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    if Gc.number_of_nodes() == 0:
        raise Exception("empty graph")
    if Gc.number_of_edges() == 0:
        raise Exception("empty edge")
    # remove_nodes = []
    # for u in Gc.nodes():
    #    if Gc
    pos_layout = nx.kamada_kawai_layout(Gc, weight=None)
    # pos_layout = nx.spring_layout(Gc, weight=None)

    weights = [d for (u, v, d) in Gc.edges(data="weight", default=1)]
    if edge_vmax is None:
        edge_vmax = statistics.median_high(
            [d for (u, v, d) in Gc.edges(data="weight", default=1)]
        )
    min_color = min([d for (u, v, d) in Gc.edges(data="weight", default=1)])
    # color range: gray to black
    edge_vmin = 2 * min_color - edge_vmax
    nx.draw(
        Gc,
        pos=pos_layout,
        with_labels=False,
        font_size=4,
        labels=feat_labels,
        node_color=node_colors,
        vmin=0,
        vmax=vmax,
        cmap=cmap,
        edge_color=edge_colors,
        edge_cmap=plt.get_cmap("Greys"),
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        width=1.0,
        node_size=50,
        alpha=0.8,
    )
    fig.axes[0].xaxis.set_visible(False)
    fig.canvas.draw()

    logdir = "log" if not hasattr(args, "logdir") or not args.logdir else str(args.logdir)
    if nodecolor != "feat":
        name += gen_explainer_prefix(args)
    save_path = os.path.join(logdir, name  + "_" + str(epoch) + ".pdf")
    print(logdir + "/" + name + gen_explainer_prefix(args) + "_" + str(epoch) + ".pdf")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format="pdf")

    img = tensorboardX.utils.figure_to_image(fig)
    writer.add_image(name, img, epoch)


def plot_cmap(cmap, ncolor):
    """
    A convenient function to plot colors of a matplotlib cmap
    Credit goes to http://gvallver.perso.univ-pau.fr/?p=712

    Args:
        ncolor (int): number of color to show
        cmap: a cmap object or a matplotlib color name
    """

    if isinstance(cmap, str):
        name = cmap
        try:
            cm = plt.get_cmap(cmap)
        except ValueError:
            print("WARNINGS :", cmap, " is not a known colormap")
            cm = plt.cm.gray
    else:
        cm = cmap
        name = cm.name

    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        fig = plt.figure(figsize=(12, 1), frameon=False)
        ax = fig.add_subplot(111)
        ax.pcolor(np.linspace(1, ncolor, ncolor).reshape(1, ncolor), cmap=cm)
        ax.set_title(name)
        xt = ax.set_xticks([])
        yt = ax.set_yticks([])
    return fig


def plot_cmap_tb(writer, cmap, ncolor, name):
    """Plot the color map used for plot."""
    fig = plot_cmap(cmap, ncolor)
    img = tensorboardX.utils.figure_to_image(fig)
    writer.add_image(name, img, 0)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

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


def read_graphfile(datadir, dataname, max_nodes=None, edge_labels=False):
    """ Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    """
    prefix = os.path.join(datadir, dataname, dataname)
    filename_graph_indic = prefix + "_graph_indicator.txt"
    # index of graphs that a given node belongs to
    graph_indic = {}
    with open(filename_graph_indic) as f:
        i = 1
        for line in f:
            line = line.strip("\n")
            graph_indic[i] = int(line)
            i += 1

    filename_nodes = prefix + "_node_labels.txt"
    node_labels = []
    min_label_val = None
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                l = int(line)
                node_labels += [l]
                if min_label_val is None or min_label_val > l:
                    min_label_val = l
        # assume that node labels are consecutive
        num_unique_node_labels = max(node_labels) - min_label_val + 1
        node_labels = [l - min_label_val for l in node_labels]
    except IOError:
        print("No node labels")

    filename_node_attrs = prefix + "_node_attributes.txt"
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [
                    float(attr) for attr in re.split("[,\s]+", line) if not attr == ""
                ]
                node_attrs.append(np.array(attrs))
    except IOError:
        print("No node attributes")

    label_has_zero = False
    filename_graphs = prefix + "_graph_labels.txt"
    graph_labels = []

    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)

    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])

    if edge_labels:
        # For Tox21_AHR we want to know edge labels
        filename_edges = prefix + "_edge_labels.txt"
        edge_labels = []

        edge_label_vals = []
        with open(filename_edges) as f:
            for line in f:
                line = line.strip("\n")
                val = int(line)
                if val not in edge_label_vals:
                    edge_label_vals.append(val)
                edge_labels.append(val)

        edge_label_map_to_int = {val: i for i, val in enumerate(edge_label_vals)}

    filename_adj = prefix + "_A.txt"
    adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
    # edge_label_list={i:[] for i in range(1,len(graph_labels)+1)}
    index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0, e1))
            index_graph[graph_indic[e0]] += [e0, e1]
            # edge_label_list[graph_indic[e0]].append(edge_labels[num_edges])
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k] = [u - 1 for u in set(index_graph[k])]

    graphs = []
    for i in range(1, 1 + len(adj_list)):
        # indexed from 1 here
        G = nx.from_edgelist(adj_list[i])

        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue

        # add features and labels
        G.graph["label"] = graph_labels[i - 1]

        # Special label for aromaticity experiment
        # aromatic_edge = 2
        # G.graph['aromatic'] = aromatic_edge in edge_label_list[i]

        for u in G.nodes():
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u - 1]
                node_label_one_hot[node_label] = 1
                G.nodes[u]["label"] = node_label_one_hot
            if len(node_attrs) > 0:
                G.nodes[u]["feat"] = node_attrs[u - 1]
        if len(node_attrs) > 0:
            G.graph["feat_dim"] = node_attrs[0].shape[0]

        # relabeling
        mapping = {}
        it = 0
        if float(nx.__version__) < 2.0:
            for n in G.nodes():
                mapping[n] = it
                it += 1
        else:
            for n in G.nodes:
                mapping[n] = it
                it += 1

        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))
    return graphs


def read_biosnap(datadir, edgelist_file, label_file, feat_file=None, concat=True):
    """ Read data from BioSnap

    Returns:
        List of networkx objects with graph and node labels
    """
    G = nx.Graph()
    delimiter = "\t" if "tsv" in edgelist_file else ","
    print(delimiter)
    df = pd.read_csv(
        os.path.join(datadir, edgelist_file), delimiter=delimiter, header=None
    )
    data = list(map(tuple, df.values.tolist()))
    G.add_edges_from(data)
    print("Total nodes: ", G.number_of_nodes())

    G = max(nx.connected_component_subgraphs(G), key=len)
    print("Total nodes in largest connected component: ", G.number_of_nodes())

    df = pd.read_csv(os.path.join(datadir, label_file), delimiter="\t", usecols=[0, 1])
    data = list(map(tuple, df.values.tolist()))

    missing_node = 0
    for line in data:
        if int(line[0]) not in G:
            missing_node += 1
        else:
            G.nodes[int(line[0])]["label"] = int(line[1] == "Essential")

    print("missing node: ", missing_node)

    missing_label = 0
    remove_nodes = []
    for u in G.nodes():
        if "label" not in G.nodes[u]:
            missing_label += 1
            remove_nodes.append(u)
    G.remove_nodes_from(remove_nodes)
    print("missing_label: ", missing_label)

    if feat_file is None:
        feature_generator = featgen.ConstFeatureGen(np.ones(10, dtype=float))
        feature_generator.gen_node_features(G)
    else:
        df = pd.read_csv(os.path.join(datadir, feat_file), delimiter=",")
        data = np.array(df.values)
        print("Feat shape: ", data.shape)

        for row in data:
            if int(row[0]) in G:
                if concat:
                    node = int(row[0])
                    onehot = np.zeros(10)
                    onehot[min(G.degree[node], 10) - 1] = 1.0
                    G.nodes[node]["feat"] = np.hstack(
                        (np.log(row[1:] + 0.1), [1.0], onehot)
                    )
                else:
                    G.nodes[int(row[0])]["feat"] = np.log(row[1:] + 0.1)

        missing_feat = 0
        remove_nodes = []
        for u in G.nodes():
            if "feat" not in G.nodes[u]:
                missing_feat += 1
                remove_nodes.append(u)
        G.remove_nodes_from(remove_nodes)
        print("missing feat: ", missing_feat)

    return G


def build_aromaticity_dataset():
    filename = "data/tox21_10k_data_all.sdf"
    basename = filename.split(".")[0]
    collector = []
    sdprovider = Chem.SDMolSupplier(filename)
    for i,mol in enumerate(sdprovider):
        try:
            moldict = {}
            moldict['smiles'] = Chem.MolToSmiles(mol)
            #Parse Data
            for propname in mol.GetPropNames():
                moldict[propname] = mol.GetProp(propname)
            nb_bonds = len(mol.GetBonds())
            is_aromatic = False; aromatic_bonds = []
            for j in range(nb_bonds):
                if mol.GetBondWithIdx(j).GetIsAromatic():
                    aromatic_bonds.append(j)
                    is_aromatic = True
            moldict['aromaticity'] = is_aromatic
            moldict['aromatic_bonds'] = aromatic_bonds
            collector.append(moldict)
        except:
            print("Molecule %s failed"%i)
    data = pd.DataFrame(collector)
    data.to_csv(basename + '_pandas.csv')


def gen_train_plt_name(args):
    return "results/" + gen_prefix(args) + ".png"


def log_assignment(assign_tensor, writer, epoch, batch_idx):
    plt.switch_backend("agg")
    fig = plt.figure(figsize=(8, 6), dpi=300)

    # has to be smaller than args.batch_size
    for i in range(len(batch_idx)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(
            assign_tensor.cpu().data.numpy()[batch_idx[i]], cmap=plt.get_cmap("BuPu")
        )
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")
    plt.tight_layout()
    fig.canvas.draw()

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    writer.add_image("assignment", data, epoch)

# TODO: unify log_graph and log_graph2
def log_graph2(adj, batch_num_nodes, writer, epoch, batch_idx, assign_tensor=None):
    plt.switch_backend("agg")
    fig = plt.figure(figsize=(8, 6), dpi=300)

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i + 1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(
            G,
            pos=nx.spring_layout(G),
            with_labels=True,
            node_color="#336699",
            edge_color="grey",
            width=0.5,
            node_size=300,
            alpha=0.7,
        )
        ax.xaxis.set_visible(False)

    plt.tight_layout()
    fig.canvas.draw()

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    writer.add_image("graphs", data, epoch)

    # log a label-less version
    # fig = plt.figure(figsize=(8,6), dpi=300)
    # for i in range(len(batch_idx)):
    #    ax = plt.subplot(2, 2, i+1)
    #    num_nodes = batch_num_nodes[batch_idx[i]]
    #    adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
    #    G = nx.from_numpy_matrix(adj_matrix)
    #    nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color='#336699',
    #            edge_color='grey', width=0.5, node_size=25,
    #            alpha=0.8)

    # plt.tight_layout()
    # fig.canvas.draw()

    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # writer.add_image('graphs_no_label', data, epoch)

    # colored according to assignment
    assignment = assign_tensor.cpu().data.numpy()
    fig = plt.figure(figsize=(8, 6), dpi=300)

    num_clusters = assignment.shape[2]
    all_colors = np.array(range(num_clusters))

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i + 1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()

        label = np.argmax(assignment[batch_idx[i]], axis=1).astype(int)
        label = label[: batch_num_nodes[batch_idx[i]]]
        node_colors = all_colors[label]

        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(
            G,
            pos=nx.spring_layout(G),
            with_labels=False,
            node_color=node_colors,
            edge_color="grey",
            width=0.4,
            node_size=50,
            cmap=plt.get_cmap("Set1"),
            vmin=0,
            vmax=num_clusters - 1,
            alpha=0.8,
        )

    plt.tight_layout()
    fig.canvas.draw()

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    writer.add_image("graphs_colored", data, epoch)
