import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx
import tensorboardX

def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
    name += '_' + args.method

    name += '_h' + str(args.hidden_dim) + '_o' + str(args.output_dim)
    if not args.bias:
        name += '_nobias'
    if len(args.name_suffix) > 0:
        name += '_' + args.name_suffix
    return name

def gen_explainer_prefix(args):
    name = gen_prefix(args) + '_explain'
    return name

def create_filename(save_dir, args, isbest=False, num_epochs=-1):
    '''
    Args:
        args: the arguments parsed in the parser
        isbest: whether the saved model is the best-performing one
        num_epochs: epoch number of the model (when isbest=False)
    '''
    filename = os.path.join(save_dir, gen_prefix(args))
    os.makedirs(filename, exist_ok=True)

    if isbest:
        filename = os.path.join(filename, 'best')
    elif num_epochs > 0:
        filename = os.path.join(filename, str(num_epochs))

    return filename + '.pth.tar'

def save_checkpoint(model, optimizer, args, num_epochs=-1, isbest=False, cg_dict=None):
    filename = create_filename(args.ckptdir, args, isbest, num_epochs=num_epochs)
    torch.save({'epoch': num_epochs,
                'model_type': args.method,
                'optimizer': optimizer,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'cg': cg_dict},
               filename)

def load_ckpt(args, isbest=False):
    print('loading model')
    filename = create_filename(args.ckptdir, args, isbest)
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        ckpt = torch.load(filename)
    return ckpt

def log_matrix(writer, mat, name, epoch, fig_size=(8,6), dpi=200):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    plt.imshow(mat.cpu().detach().numpy(), cmap=plt.get_cmap('BuPu'))
    cbar = plt.colorbar()
    cbar.solids.set_edgecolor("face")

    plt.tight_layout()
    fig.canvas.draw()
    writer.add_image(name, tensorboardX.utils.figure_to_image(fig), epoch)

def denoise_graph(adj, node_idx, feat=None, label=None, threshold=0.1):
    num_nodes = adj.shape[-1]
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.node[node_idx]['self'] = 1
    #print('num nodes : ', G.number_of_nodes())
    if feat is not None:
        for node in G.nodes():
            G.node[node]['feat'] = feat[node]
    if label is not None:
        for node in G.nodes():
            G.node[node]['label'] = label[node]
    weighted_edge_list = [(i, j, adj[i, j]) for i in range(num_nodes) for j in range(num_nodes) if
            adj[i,j] > threshold]
    G.add_weighted_edges_from(weighted_edge_list)
    Gc = max(nx.connected_component_subgraphs(G), key=len) 
    return Gc

def log_graph(writer, Gc, name, epoch=0, fig_size=(4,3), dpi=300):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=fig_size, dpi=dpi)
   
    node_colors = []
    edge_colors = [min(max(w, 0.0), 1.0) for (u,v,w) in Gc.edges.data('weight', default=1)]
    for i in Gc.nodes():
        if 'self' in Gc.node[i]:
            node_colors.append(0)
        elif 'label' in Gc.node[i]:
            node_colors.append(Gc.node[i]['label'] + 1)
        else:
            node_colors.append(1)

    plt.switch_backend('agg')
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    nx.draw(Gc, pos=nx.spring_layout(Gc), with_labels=True, font_size=4,
            node_color=node_colors, vmin=0, vmax=8, cmap=plt.get_cmap('Set1'),
            edge_color=edge_colors, edge_cmap=plt.get_cmap('Greys'), edge_vmin=0.0, edge_vmax=1.0,
            width=0.5, node_size=25,
            alpha=0.7)
    fig.axes[0].xaxis.set_visible(False)
    fig.canvas.draw()
    plt.savefig('log/' + name)
    img = tensorboardX.utils.figure_to_image(fig)
    writer.add_image(name, img, epoch)


