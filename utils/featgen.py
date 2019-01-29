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
