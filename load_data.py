"""load_data.py

   Utilities for data loading.
"""

import os
import re
import csv

import networkx as nx
import numpy as np
import pandas as pd
import scipy as sc

import utils.featgen as featgen

def read_graphfile(datadir, dataname, max_nodes = None, edge_labels = False):
    ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    '''
    prefix = os.path.join(datadir, dataname, dataname)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic={}
    with open(filename_graph_indic) as f:
        i=1
        for line in f:
            line=line.strip("\n")
            graph_indic[i]=int(line)
            i+=1

    filename_nodes=prefix + '_node_labels.txt'
    node_labels=[]
    min_label_val = None
    try:
        with open(filename_nodes) as f:
            for line in f:
                line=line.strip("\n")
                l = int(line)
                node_labels+=[l]
                if min_label_val is None or min_label_val > l:
                    min_label_val = l
        # assume that node labels are consecutive
        num_unique_node_labels = max(node_labels) - min_label_val + 1
        node_labels = [l - min_label_val for l in node_labels]
    except IOError:
        print('No node labels')
 
    filename_node_attrs=prefix + '_node_attributes.txt'
    node_attrs=[]
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')
       
    label_has_zero = False
    filename_graphs=prefix + '_graph_labels.txt'
    graph_labels=[]

    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line=line.strip("\n")
            val = int(line)
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)

    label_map_to_int = {val:i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])

    if edge_labels:
        # For Tox21_AHR we want to know edge labels 
        filename_edges = prefix + '_edge_labels.txt'
        edge_labels = []

        edge_label_vals = []
        with open(filename_edges) as f:
            for line in f:
                line=line.strip("\n")
                val = int(line)
                if val not in edge_label_vals:
                    edge_label_vals.append(val)
                edge_labels.append(val)

        edge_label_map_to_int = {val:i for i, val in enumerate(edge_label_vals)}

    filename_adj=prefix + '_A.txt'
    adj_list={i:[] for i in range(1,len(graph_labels)+1)}    
    #edge_label_list={i:[] for i in range(1,len(graph_labels)+1)}
    index_graph={i:[] for i in range(1,len(graph_labels)+1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line=line.strip("\n").split(",")
            e0,e1=(int(line[0].strip(" ")),int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0,e1))
            index_graph[graph_indic[e0]]+=[e0,e1]
            #edge_label_list[graph_indic[e0]].append(edge_labels[num_edges])
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k]=[u-1 for u in set(index_graph[k])]

    graphs=[]
    for i in range(1,1+len(adj_list)):
        # indexed from 1 here
        G=nx.from_edgelist(adj_list[i])
        
        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue
      
        # add features and labels
        G.graph['label'] = graph_labels[i-1]
        
        # Special label for aromaticity experiment
        # aromatic_edge = 2
        # G.graph['aromatic'] = aromatic_edge in edge_label_list[i]

        for u in G.nodes():
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u-1]
                node_label_one_hot[node_label] = 1
                G.node[u]['label'] = node_label_one_hot
            if len(node_attrs) > 0:
                G.node[u]['feat'] = node_attrs[u-1]
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]

        # relabeling
        mapping={}
        it=0
        if float(nx.__version__)<2.0:
            for n in G.nodes():
                mapping[n]=it
                it+=1
        else:
            for n in G.nodes:
                mapping[n]=it
                it+=1
            
        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))
    return graphs


def read_biosnap(datadir, edgelist_file, label_file, feat_file=None, concat=True):
    ''' Read data from BioSnap

    Returns:
        List of networkx objects with graph and node labels
    '''
    G = nx.Graph()
    delimiter = '\t' if 'tsv' in edgelist_file else ','
    print(delimiter)
    df = pd.read_csv(os.path.join(datadir, edgelist_file), delimiter=delimiter, header=None,)
    data = list(map(tuple, df.values.tolist()))
    G.add_edges_from(data)
    print('Total nodes: ', G.number_of_nodes())

    G = max(nx.connected_component_subgraphs(G), key=len)
    print('Total nodes in largest connected component: ', G.number_of_nodes())

    df = pd.read_csv(os.path.join(datadir, label_file), delimiter='\t', usecols=[0, 1])
    data = list(map(tuple, df.values.tolist()))

    missing_node = 0
    for line in data:
        if int(line[0]) not in G:
            missing_node += 1
        else:
            G.node[int(line[0])]['label'] = int(line[1] == 'Essential')

    print('missing node: ', missing_node)
    
    missing_label = 0
    remove_nodes = []
    for u in G.nodes():
        if 'label' not in G.node[u]:
            missing_label += 1
            remove_nodes.append(u)
    G.remove_nodes_from(remove_nodes)
    print('missing_label: ', missing_label)

    if feat_file is None:
        feature_generator=featgen.ConstFeatureGen(np.ones(10, dtype=float))
        feature_generator.gen_node_features(G)
    else:
        df = pd.read_csv(os.path.join(datadir, feat_file), delimiter=',')
        data = np.array(df.values)
        print('Feat shape: ', data.shape)

        for row in data:
            if int(row[0]) in G:
                if concat:
                    node = int(row[0])
                    onehot = np.zeros(10)
                    onehot[min(G.degree[node], 10) - 1] = 1.
                    G.node[node]['feat'] = np.hstack((np.log(row[1:] + 0.1), 
                            [1.], onehot))
                else:
                    G.node[int(row[0])]['feat'] = np.log(row[1:] + 0.1)

        missing_feat = 0
        remove_nodes = []
        for u in G.nodes():
            if 'feat' not in G.node[u]:
                missing_feat += 1
                remove_nodes.append(u)
        G.remove_nodes_from(remove_nodes)
        print('missing feat: ', missing_feat)
   
    return G


