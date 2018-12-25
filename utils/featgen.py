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
        feat_dict = {i:{'feat': self.val} for i in G.nodes()}
        nx.set_node_attributes(G, feat_dict)

class GaussianFeatureGen(FeatureGen):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def gen_node_features(self, G):
        feat = np.random.multivariate_normal(mu, sigma, G.number_of_nodes())
        feat_dict = {i:{'feat': feat[i]} for i in range(feat.shape[0])}
        nx.set_node_attributes(G, feat_dict)
