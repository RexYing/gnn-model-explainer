import abc
import networkx as nx
import numpy as np
import random

class FeatureGen(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def gen_node_features(self, G):
        pass

class ConstFeatureGen(FeatureGen):
    def __init__(self, val):
        self.val = val

    def gen_node_features(self, G):
        feat_dict = {i:{'feat': np.array(self.val, dtype=np.float32)} for i in G.nodes()}
        # print ('feat_dict[0]["feat"]:', feat_dict[0]['feat'].dtype)
        nx.set_node_attributes(G, feat_dict)
        # print ('G.node[0]["feat"]:', G.node[0]['feat'].dtype)

class GaussianFeatureGen(FeatureGen):
    def __init__(self, mu, sigma):
        self.mu = mu
        if sigma.ndim < 2:
            self.sigma = np.diag(sigma)
        else:
            self.sigma = sigma

    def gen_node_features(self, G):
        feat = np.random.multivariate_normal(self.mu, self.sigma, G.number_of_nodes())
        feat_dict = {i:{'feat': np.array(feat[i], dtype=np.float32)} for i in range(feat.shape[0])}
        nx.set_node_attributes(G, feat_dict)

class GridFeatureGen(FeatureGen):
    def __init__(self, mu, sigma, com_choices):
        self.mu = mu
        self.sigma = sigma
        self.com_choices = com_choices

    def gen_node_features(self, G):
        # Generate community assignment
        community_dict = {n: self.com_choices[0] if G.degree(n) < 4 else self.com_choices[1] for n in G.nodes()}
        print(community_dict)
        # Generate random variable
        s = np.random.normal(self.mu, self.sigma, G.number_of_nodes())

        # Generate features
        feat_dict = {n:{'feat': np.asarray([community_dict[n], s[i]])} for i,n in enumerate(G.nodes())}

        nx.set_node_attributes(G, feat_dict)
        return community_dict

