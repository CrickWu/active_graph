import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from utils import convert_edge2adj, normalize

# Factory class:
class ActiveFactory:
    def __init__(self, args, model, data):
        # 
        self.args = args
        self.model = model
        self.data = data

    def get_learner(self):
        if self.args.method == 'random':
            self.learner = RandomLearner
        elif self.args.method == 'kmeans':
            self.learner = KmeansLearner
        elif self.args.method == 'degree':
            self.learner = DegreeLearner
        return self.learner(self.args, self.model, self.data)

# Base class
class ActiveLearner:
    def __init__(self, args, model, data):
        self.model = model
        self.data = data
        self.n = data.num_nodes
        self.args = args
        self.adj_full = convert_edge2adj(data.edge_index, data.num_nodes)

    def choose(self, num_points):
        raise NotImplementedError

    def pretrain_choose(self, num_points):
        raise NotImplementedError


class KmeansLearner(ActiveLearner):
    def __init__(self, args, model, data):
        super(KmeansLearner, self).__init__(args, model, data)
        self.device = data.x.get_device()
        self.norm_adj = normalize(self.adj_full).to(self.device)

    def pretrain_choose(self, num_points):
        features = self.data.x
        for k in range(self.args.kmeans_num_layer):
            features = self.norm_adj.matmul(features)
        features = features.cpu().numpy()

        kmeans = KMeans(n_clusters=num_points).fit(features)
        center_dist = pairwise_distances(kmeans.cluster_centers_, features) # k x n
        indices = torch.LongTensor(np.argmin(center_dist, axis=1))
        ret_tensor = torch.zeros((self.n), dtype=torch.uint8)
        ret_tensor[indices] = 1
        return ret_tensor

class RandomLearner(ActiveLearner):
    def __init__(self, args, model, data):
        super(RandomLearner, self).__init__(args, model, data)
    def pretrain_choose(self, num_points):
        return torch.multinomial(torch.range(start=0, end=self.n-1), num_samples=num_points, replacement=False)

class DegreeLearner(ActiveLearner):
    def __init__(self, args, model, data):
        super(DegreeLearner, self).__init__(args, model, data)
    def pretrain_choose(self, num_points):
        ret_tensor = torch.zeros((self.n), dtype=torch.uint8)
        degree_full = self.adj_full.sum(dim=1)
        vals, indices = torch.topk(degree_full, k=num_points)
        ret_tensor[indices] = 1
        return ret_tensor

# impose all category constraint
# no direct linkage
class NonOverlapDegreeLearner(ActiveLearner):
    def __init__(self, args, model, data):
        super(NonOverlapDegreeLearner, self).__init__(args, model, data)
    def pretrain_choose(self, num_points):
        # select by degree
        ret_tensor = torch.zeros((self.n), dtype=torch.uint8)
        degree_full = self.adj_full.sum(dim=1)
        vals, indices = torch.sort(degree_full, descending=True)
        
        index_list = []
        
        num = 0
        for i in indices:
            edge_flag = False
            for j in index_list:
                if self.adj_full[i, j] != 0:
                    edge_flag = True
                    break
            if not edge_flag:
                index_list.append(i)
                num += 1
            if num == num_points:
                break
        
        ret_tensor[torch.LongTensor(index_list)] = 1
        return ret_tensor