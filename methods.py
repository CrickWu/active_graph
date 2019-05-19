import numpy as np
import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from utils import convert_edge2adj, normalize
from utils import kcenter_choose, kmeans_choose, combine_new_old

import time

# Factory class:
class ActiveFactory:
    def __init__(self, args, model, data, prev_index):
        # 
        self.args = args
        self.model = model
        self.data = data
        self.prev_index = prev_index

    def get_learner(self):
        if self.args.method == 'random':
            self.learner = RandomLearner
        elif self.args.method == 'kmeans':
            self.learner = KmeansLearner
        elif self.args.method == 'degree':
            self.learner = DegreeLearner
        elif self.args.method == 'nonoverlapdegree':
            self.learner = NonOverlapDegreeLearner
        elif self.args.method == 'coreset':
            self.learner = CoresetLearner
        elif self.args.method == 'uncertain':
            self.learner = UncertaintyLearner
        return self.learner(self.args, self.model, self.data, self.prev_index)

# Base class
class ActiveLearner:
    def __init__(self, args, model, data, prev_index):
        self.model = model
        self.data = data
        self.n = data.num_nodes
        self.args = args
        self.prev_index = prev_index
        
        if prev_index is None:
            self.prev_index_list = []
        else:
            self.prev_index_list = np.where(self.prev_index.cpu().numpy())[0]

        start = time.time()
        self.adj_full = convert_edge2adj(data.edge_index, data.num_nodes)
        print('Time cost: {}'.format(time.time() - start))

    def choose(self, num_points):
        raise NotImplementedError

    def pretrain_choose(self, num_points):
        raise NotImplementedError

class UncertaintyLearner(ActiveLearner):
    def __init__(self, args, model, data, prev_index):
        super(UncertaintyLearner, self).__init__(args, model, data, prev_index)
        self.device = data.x.get_device()

    def pretrain_choose(self, num_points):
        self.model.eval()
        (features, prev_out, no_softmax), out = self.model(self.data)

        if self.args.uncertain_score == 'entropy':
            scores = torch.sum(-F.softmax(prev_out, dim=1) * F.log_softmax(prev_out, dim=1), dim=1)
        elif self.args.uncertain_score == 'margin':
            pred = F.softmax(prev_out, dim=1)
            top_pred, _ = torch.topk(pred, k=2, dim=1)
            # use negative values, since the largest values will be chosen as labeled data
            scores =  (-top_pred[:,0] + top_pred[:,1]).view(-1)
        else:
            raise NotImplementedError


        vals, full_new_index_list = torch.topk(scores, k=num_points)
        full_new_index_list = full_new_index_list.cpu().numpy()

        '''

        # excluding existing indices
        add_index_list = []
        exist_num = 0
        for cur_index in new_index_list:
            if cur_index not in self.prev_index_list:
                exist_num += 1
                add_index_list.append(cur_index)
            if exist_num == num_points - len(self.prev_index_list):
                break

        indices = torch.LongTensor( np.concatenate((self.prev_index_list, add_index_list)) )
        ret_tensor = torch.zeros((self.n), dtype=torch.uint8)
        ret_tensor[indices] = 1
        return ret_tensor'''
        return combine_new_old(full_new_index_list, self.prev_index_list, num_points, self.n, in_order=True)

class CoresetLearner(ActiveLearner):
    def __init__(self, args, model, data, prev_index):
        super(CoresetLearner, self).__init__(args, model, data, prev_index)
        self.device = data.x.get_device()

    def pretrain_choose(self, num_points):
        # random selection if the model is untrained
        if self.prev_index is None:
            return torch.multinomial(torch.range(start=0, end=self.n-1), num_samples=num_points, replacement=False)

        self.model.eval()
        (features, prev_out, no_softmax), out = self.model(self.data)

        features = features.cpu().detach().numpy()
        '''
        # TODO: should be modified to K-center method
        kmeans = KMeans(n_clusters=num_points).fit(features)
        center_dist = pairwise_distances(kmeans.cluster_centers_, features) # k x n

        new_index_list = np.argmin(center_dist, axis=1)
        prev_index_len = len(self.prev_index_list)
        diff_list = np.asarray(list(set(new_index_list).difference(set(self.prev_index_list))))
        indices = torch.LongTensor( np.concatenate((self.prev_index_list, diff_list[:-prev_index_len + num_points])) )
        ret_tensor = torch.zeros((self.n), dtype=torch.uint8)
        ret_tensor[indices] = 1
        '''
        if self.args.cluster_method == 'kmeans':
            return kmeans_choose(features, num_points, prev_index_list=self.prev_index_list, n=self.n)
        elif self.args.cluster_method == 'kcenter':
            return kcenter_choose(features, num_points, prev_index_list=self.prev_index_list, n=self.n)
        else:
            raise NotImplementedError

class KmeansLearner(ActiveLearner):
    def __init__(self, args, model, data, prev_index):
        super(KmeansLearner, self).__init__(args, model, data, prev_index)
        self.device = data.x.get_device()
        self.norm_adj = normalize(self.adj_full + torch.eye(self.n) * self.args.self_loop_coeff).to(self.device)

    def pretrain_choose(self, num_points):
        features = self.data.x
        for k in range(self.args.kmeans_num_layer):
            features = self.norm_adj.matmul(features)
        features = features.cpu().numpy()

        # Note all prev_index_list's are empty since features of KmeansLearner do not rely on previous results, and have not clue of the intermediate model status (it trains from scratch)
        if self.args.cluster_method == 'kmeans':
            return kmeans_choose(features, num_points, prev_index_list=[], n=self.n)
        elif self.args.cluster_method == 'kcenter':
            return kcenter_choose(features, num_points, prev_index_list=[], n=self.n)
        else:
            raise NotImplementedError
        '''
        kmeans = KMeans(n_clusters=num_points).fit(features)
        center_dist = pairwise_distances(kmeans.cluster_centers_, features) # k x n
        indices = torch.LongTensor(np.argmin(center_dist, axis=1))
        ret_tensor = torch.zeros((self.n), dtype=torch.uint8)
        ret_tensor[indices] = 1
        return ret_tensor
        '''

class RandomLearner(ActiveLearner):
    def __init__(self, args, model, data, prev_index):
        super(RandomLearner, self).__init__(args, model, data, prev_index)
    def pretrain_choose(self, num_points):
        return torch.multinomial(torch.range(start=0, end=self.n-1), num_samples=num_points, replacement=False)

class DegreeLearner(ActiveLearner):
    def __init__(self, args, model, data, prev_index):
        super(DegreeLearner, self).__init__(args, model, data, prev_index)
    def pretrain_choose(self, num_points):
        ret_tensor = torch.zeros((self.n), dtype=torch.uint8)
        degree_full = self.adj_full.sum(dim=1)
        vals, indices = torch.topk(degree_full, k=num_points)
        ret_tensor[indices] = 1
        return ret_tensor

# impose all category constraint
# no direct linkage
class NonOverlapDegreeLearner(ActiveLearner):
    def __init__(self, args, model, data, prev_index):
        super(NonOverlapDegreeLearner, self).__init__(args, model, data, prev_index)
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