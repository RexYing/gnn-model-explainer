import os

import matplotlib.pyplot as plt
import torch
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

def log_graph(writer, adj, name, epoch=0, fig_size=(4,3), dpi=300):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    
    num_nodes = adj.size()[-1]
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.node[node_idx]['color'] = 0
    weighted_edge_list = [(i, j, adj[i, j]) for i in range(num_nodes) for j in range(num_nodes) if adj[i,j] > 0.1]
    G.add_weighted_edges_from(weighted_edge_list)
    Gc = max(nx.connected_component_subgraphs(G), key=len)
    edge_colors = [Gc[i][j]['weight'] for (i,j) in Gc.edges()]
    node_colors = [Gc.node[i]['color'] if 'color' in Gc.node[i] else 1 for i in Gc.nodes()]

    plt.switch_backend('agg')
    fig = plt.figure(figsize=(4,3), dpi=600)
    nx.draw(Gc, pos=nx.spring_layout(G), with_labels=True, font_size=4,
            node_color=node_colors, vmin=0, vmax=8, cmap=plt.get_cmap('Set1'),
            edge_color=edge_colors, edge_cmap=plt.get_cmap('Greys'), edge_vmin=0.0, edge_vmax=1.0,
            width=0.5, node_size=25,
            alpha=0.7)
    fig.axes[0].xaxis.set_visible(False)
    fig.canvas.draw()
    self.writer.add_image(name, tensorboardX.utils.figure_to_image(fig), epoch)


