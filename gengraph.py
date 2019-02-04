from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import os
import networkx as nx
import numpy as np
from tensorboardX import SummaryWriter
import utils.io_utils as io_utils

from utils import synthetic_structsim
from utils import featgen

plt.switch_backend('agg')

def perturb_new(graph_list, p):
    ''' Perturb the list of graphs by adding/removing edges.
    Args:
        p_add: probability of adding edges. If None, estimate it according to graph density,
            such that the expected number of added edges is equal to that of deleted edges.
        p_del: probability of removing edges
    Returns:
        A list of graphs that are perturbed from the original graphs
    '''
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        #edge_remove_count = 0
        #for (u, v) in list(G.edges()):
        #    if np.random.rand()<p:
        #        G.remove_edge(u, v)
        #        edge_remove_count += 1
        edge_remove_count = int(G.number_of_edges() * p)
        # randomly add the edges back
        for i in range(edge_remove_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u,v)) and (u!=v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list

   
def join_graph(G1, G2, n_pert_edges):
    F = nx.compose(G1,G2)  
    n_edges = 0
    while n_edges < n_pert_edges:
        node_1 = np.random.choice(G1.nodes())
        node_2 = np.random.choice(G2.nodes())
        F.add_edge(node_1, node_2)
        n_edges += 1
    return F 

#####
#
# Generate input graph
#
#####
def gen_syn1(nb_shapes = 80, width_basis = 300, feature_generator=None, m=5):
    basis_type = 'ba'
    list_shapes = [['house']] * nb_shapes

    fig = plt.figure(figsize=(8,6), dpi=300)

    G, role_id, plugins = synthetic_structsim.build_graph(width_basis, basis_type, list_shapes,
            start=0, m=5)
    G = perturb_new([G], 0.01)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + '_' + str(width_basis) + '_' + str(nb_shapes)
    return G, role_id, name


def gen_syn2(nb_shapes = 100, width_basis = 350):
    basis_type = 'ba'

    random_mu = [0.0] * 8
    random_sigma = [1.0] * 8
    # Create two grids
    #mu_1, sigma_1 = np.array([0.05 * i for i in range(10)]), np.array([0.5] * 10)
    #mu_2, sigma_2 = np.array([1 - 0.05 * i for i in range(10)]), np.array([0.5] * 10)
    mu_1, sigma_1 = np.array([-1.0]*2 + random_mu), np.array([0.5]*2 + random_sigma)
    mu_2, sigma_2 = np.array([1.0]*2 + random_mu), np.array([0.5]*2 + random_sigma)
    feat_gen_G1 = featgen.GaussianFeatureGen(mu=mu_1, sigma=sigma_1)
    feat_gen_G2 = featgen.GaussianFeatureGen(mu=mu_2, sigma=sigma_2)
    G1, role_id1, name = gen_syn1(feature_generator=feat_gen_G1, m=4)
    G2, role_id2, name = gen_syn1(feature_generator=feat_gen_G2, m=4)
    G1_size = G1.number_of_nodes()
    num_roles = max(role_id1) + 1
    role_id2 = [r + num_roles for r in role_id2]
    label = role_id1 + role_id2
    
    #G1 = nx.grid_graph(dim=[N,N])
    #G2 = nx.grid_graph(dim=[N,N])
    # Edit node ids to avoid collisions on join
    g1_map = {n:i for i,n in enumerate(G1.nodes())}
    G1 = nx.relabel_nodes(G1, g1_map)
    g2_map = {n:i+G1_size for i,n in enumerate(G2.nodes())}
    G2 = nx.relabel_nodes(G2, g2_map)

    # Create node features
    #com_choices_1 = [0,1]
    #feat_gen_G1 = featgen.GridFeatureGen(mu=mu_1, sigma=sigma_1, com_choices=com_choices_1)
    #communities_G1 = feat_gen_G1.gen_node_features(G1) 

    #com_choices_2 = [2,3]
    #feat_gen_G2 = featgen.GridFeatureGen(mu=mu_2, sigma=sigma_2, com_choices=com_choices_2)
    #communities_G2 = feat_gen_G2.gen_node_features(G2) 

    # Join
    #n_pert_edges = int(np.log(N)) + 1; 
    n_pert_edges = width_basis
    G = join_graph(G1, G2, n_pert_edges)
   
    #communities = {**communities_G1, **communities_G2}
    #communities = list(communities.values())
    #print(label)
    #name = basis_type + '_N_' + str(N) + '_classes_' + str(n_classes) + '_pert_' + str(n_pert_edges)
    name = basis_type + '_' + str(width_basis) + '_' + str(nb_shapes) + '_2comm'

    return G, label, name

def gen_syn3(nb_shapes = 80, width_basis = 300, feature_generator=None, m=5):
    basis_type = 'ba'
    list_shapes = [['cycle', 5]] * nb_shapes

    fig = plt.figure(figsize=(8,6), dpi=300)

    G, role_id, plugins = synthetic_structsim.build_graph(width_basis, basis_type, list_shapes,
            start=0, m=5)
    G = perturb_new([G], 0.01)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + '_' + str(width_basis) + '_' + str(nb_shapes)
    return G, role_id, name


def gen_syn4(nb_shapes = 60, width_basis = 8, feature_generator=None, m=4):
    basis_type = 'tree'
    list_shapes = [['cycle', 6]] * nb_shapes

    fig = plt.figure(figsize=(8,6), dpi=300)

    G, role_id, plugins = synthetic_structsim.build_graph(width_basis, basis_type, list_shapes,
            start=0)
    G = perturb_new([G], 0.01)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + '_' + str(width_basis) + '_' + str(nb_shapes)

    path = os.path.join('log/syn4_base_h20_o20')
    writer = SummaryWriter(path)
    io_utils.log_graph(writer, G, 'graph/full')

    return G, role_id, name

def gen_syn5(nb_shapes = 80, width_basis = 8, feature_generator=None, m=3):
    basis_type='tree'
    list_shapes = [['grid', m]] * nb_shapes
    fig = plt.figure(figsize=(8,6), dpi=300)

    G, role_id, plugins = synthetic_structsim.build_graph(width_basis, basis_type, list_shapes,
            start=0)
    G = perturb_new([G], 0.1)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + '_' + str(width_basis) + '_' + str(nb_shapes)

    path = os.path.join('log/syn5_base_h20_o20')
    writer = SummaryWriter(path)
    #io_utils.log_graph(writer, G, 'graph/full')

    return G, role_id, name



def preprocess_input_graph(G, labels, normalize_adj=False):
    adj = np.array(nx.to_numpy_matrix(G))
    if normalize_adj:
        sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
        adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)

    feat_dim = G.node[0]['feat'].shape[0]
    f = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        f[i, :] = G.node[u]['feat']

    # add batch dim
    adj = np.expand_dims(adj, axis=0)
    f = np.expand_dims(f, axis=0)
    labels = np.expand_dims(labels, axis=0)
    return {'adj': adj, 'feat': f, 'labels': labels}

if __name__ == "__main__":
    #G, labels, name = gen_syn1(feature_generator=featgen.ConstFeatureGen(np.ones(5, dtype=float)))
    G, labels, name = gen_syn2()

    fig = plt.figure(figsize=(8,6), dpi=300)
    labels_dict = {i:labels[i] for i in range(G.number_of_nodes()) if not labels[i]==0}
    nx.draw(G, pos=nx.kamada_kawai_layout(G), node_size=50, node_color='#336699',
            labels=labels_dict, font_size=8,
            edge_color='grey', width=0.5, alpha=0.7)
    fig.canvas.draw()

    plt.savefig('syn/graph_' + name)
    plt.close()

