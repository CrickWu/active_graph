import numpy as np
import argparse
import os
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F
import time

from torch_geometric.datasets import Planetoid, PPI, Amazon, CoraFull

import methods
from methods import ActiveFactory
from models import get_model

from query_methods import CoreSetSampling, CoreSetMIPSampling
from utils import normalize, convert_edge2adj
from metrics import final_eval, METRIC_NAMES


# Network definition, could be refactored
# class Net(torch.nn.Module):
#     def __init__(self, args, data):
#         super(Net, self).__init__()
#         self.conv1 = GCNConv(args.num_features, args.hid_dim)
#         self.conv2 = GCNConv(args.hid_dim, args.num_classes)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index)
#         hid_x = F.relu(x)
#         x = F.dropout(hid_x, training=self.training)
#         x = self.conv2(x, edge_index)

#         return (hid_x, x), F.log_softmax(x, dim=1)

# Tool functions
def eval_model(model, data, test_mask):
    model.eval()
    _, pred = model(data)[1].max(dim=1)
    correct = pred[test_mask].eq(data.y[test_mask]).sum().item()
    acc = correct / test_mask.sum().item()
    model.train()
    return acc

def eval_model_f1(model, data, data_y, test_mask):
    model.eval()
    # TODO: whehther transform it into int?
    # pred = model(data)[0][2] > 0. # without sigmoid
    # DEBUG: why the following line is all zero while the previous is not ????
    pred = model(data)[1] > 0. # without sigmoid
    # micro F1
    correct = (pred[test_mask] & data_y[test_mask]).sum().item() # TP
    prec = correct / pred[test_mask].sum().item()
    rec = correct / data_y[test_mask].sum().item()
    model.train()
    # TODO: check correctness
    # micro_F1 = correct / test_mask.sum().item()  # precion / recall
    return 2 * prec * rec / (prec + rec)




# argparse
parser = argparse.ArgumentParser(description='Active graph learning.')
parser.add_argument('--dataset', type=str, default='Cora',
                    help='dataset used')
parser.add_argument('--label_list', type=int, nargs='+', default=[10, 20, 40, 80],
                    help='#labeled training data')
# verbose
parser.add_argument('--verbose', action='store_true',
                    help='verbose mode for training and debugging')

# random seed and optimization
parser.add_argument('--seed', type=int, default=123,
                    help='random seed for reproduction')
parser.add_argument('--epoch', type=int, default=40,
                    help='training epoch for each training setting （fixed number of training data')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='learning rate')

# GCN parameters
parser.add_argument('--hid_dim', type=int, default=16,
                    help='hidden dimension for GCN')

# Active method
parser.add_argument('--model', type=str, default='GCN',
                    help='back-end classifier, choose from [GCN, MatrixGCN, SGC]')
parser.add_argument('--method', type=str, default='random',
                    help='choice between [random, kmeans, ...]')
parser.add_argument('--rand_rounds', type=int, default=1,
                    help='number of rounds for averaging scores')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='prob of an element to be zeroed.')

### Options within method 
# KMeans
parser.add_argument('--kmeans_num_layer', type=int, default=0,
                    help='number of propergation for KMeans to generate new features')
parser.add_argument('--self_loop_coeff', type=float, default=0.,
                    help='self-loop coefficient when performing random walk convolution')

# Uncertainty options
parser.add_argument('--uncertain_score', type=str, default='entropy',
                    help='choice between [entropy, margin]')

# clustering method
parser.add_argument('--cluster_method', type=str, default='kmeans',
                    help='clustering method in kmeans and coreset; choice between [kmeans, kcenter]')
####

# dataset parsed info; usually not manually specified
parser.add_argument('--num_features', type=int, default=None,
                    help='initial feature dimension for input dataset')
parser.add_argument('--num_classes', type=int, default=None,
                    help='number of classes for node classification')
parser.add_argument('--multilabel', action='store_true',
                    help='whether the output is multi-label')

# TODO: replace with the pseudo-command line
args = parser.parse_args()

# preprocessing of data and model

# device specification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.dataset[:3] == 'PPI':
    args.multilabel = True
    dataset = PPI(root='./data/PPI')
    dataset_num = int(args.dataset[3:])
    data = dataset[dataset_num].to(device)
elif args.dataset in ['Cora', 'Citeseer', 'PubMed']:
    dataset = Planetoid(root='./data/{}'.format(args.dataset), name='{}'.format(args.dataset))
    data = dataset[0].to(device)
elif args.dataset in ['Computers', 'Photo']:
    dataset = Amazon(root='./data/{}'.format(args.dataset), name='{}'.format(args.dataset))
    data = dataset[0].to(device)
elif args.dataset in ['CoraFull']:
    dataset = CoraFull(root='./data/{}'.format(args.dataset))
    data = dataset[0].to(device)
else:
    raise NotImplementedError
Net = get_model(args.model)


args.num_features = dataset.num_features 
args.num_classes = dataset.num_classes

print(args)

# 2 types of AL
# - 1. fresh start of optimizer and model
# - 2. fresh start of optimizer and NOT model

# TODO: should consider interactive selection of nodes
def active_learn(k, data, old_model, old_optimizer, prev_index, args):
    if args.multilabel:
        loss_func = torch.nn.BCEWithLogitsLoss()
    else:
        loss_func = F.nll_loss
    test_mask = torch.ones(data.y.shape[0], dtype=torch.uint8)
    data_y = data.y > 0.99 # cast to uint8 for downstream-fast computation
    # test_mask = torch.ones_like(data.test_mask)
    # for multi-class
    num_class = torch.unique(data.y).shape[0]

    # DEBUG: unify the writing system
    if args.method in ['xcoreset', 'xcoresetmip', 'mip']:
        if prev_index is None:
            # return random-seeds
            learner = methods.RandomLearner(args, old_model, data, prev_index)
            train_mask = learner.pretrain_choose(k)
        else:
            if args.method == 'xcoreset':
                learner = CoreSetSampling(old_model, input_shape=None, num_labels=args.num_classes,gpu=1)
            elif args.method == 'xcoresetmip':
                learner = CoreSetMIPSampling(old_model, input_shape=None, num_labels=args.num_classes,gpu=1)
            elif args.method == 'mip':
                adj_full = convert_edge2adj(data.edge_index, data.num_nodes)
                norm_adj = normalize(adj_full + torch.eye(data.num_nodes) * args.self_loop_coeff).to(device)
                features = data.x
                for k in range(args.kmeans_num_layer):
                    features = norm_adj.matmul(features)
                features = features.cpu().numpy()
                learner = CoreSetMIPSampling(old_model, input_shape=None, num_labels=args.num_classes,gpu=1)

            train_mask = torch.zeros(data.y.shape[0], dtype=torch.uint8)
            prev_index_list = np.where(prev_index.cpu().numpy())[0]
            if args.method == 'mip':
                train_mask_list = torch.LongTensor(learner.query(data, prev_index_list, k-len(prev_index_list), representation=features))
            else:
                train_mask_list = torch.LongTensor(learner.query(data, prev_index_list, k-len(prev_index_list)))
            train_mask[train_mask_list] = 1
    else:
        learner = ActiveFactory(args, old_model, data, prev_index).get_learner()
        train_mask = learner.pretrain_choose(k)

    model = Net(args, data)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    
    if args.verbose:
        print('selected labels:', data.y[train_mask])
    # fresh new training
    for epoch in tqdm(range(args.epoch)):
        # Optimize GCN
        optimizer.zero_grad()
        _, out = model(data)
        # loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss = loss_func(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        # here we compute multiple measurements
        if args.multilabel:
            acc = eval_model_f1(model, data, data_y, test_mask)
        else:
            acc = eval_model(model, data, test_mask)
        if args.verbose:
            print('epoch {} acc: {:.4f} loss: {:.4f}'.format(epoch, acc, loss.item()))
    # compute all metrics in the final round
    all_metrics = final_eval(model, data, test_mask, num_class) # acc, macro_f1
    return all_metrics, train_mask, model, optimizer
    # return acc, train_mask, model, optimizer


# res = np.zeros((args.rand_rounds, len(args.label_list)))
res = [[None for j in range(len(args.label_list))] for i in range(args.rand_rounds)]
print('Using', device, 'for neural network training')
# record corresponding y labels
# record corresponding x instances

y_label = []
x_label = []
start_time = time.time()

metric_names = METRIC_NAMES
# different random seeds
for num_round in range(args.rand_rounds):
    train_mask = None
    # here should be initialized with different seeds
    torch.manual_seed(num_round+args.seed)  # for GPU and CPU after torch 1.0
    np.random.seed(num_round+args.seed)
    model = Net(args, data)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    single_y_label = []
    single_x_label = []
    # for some methods, the current selection is dependent on previous results
    for num, k in enumerate(args.label_list):
        # lr should be 0.001??
        # replace old model, optimizer with new model
        # all_metrics is a tuple
        all_metrics, train_mask, model, optimizer = active_learn(k, data, model, optimizer, train_mask, args)
        single_x_label.append(np.where(train_mask.cpu().numpy())[0].tolist())
        single_y_label.append(data.y[single_x_label[-1]].cpu().numpy().tolist())

        res[num_round][num] = all_metrics
        metric_format = ' '.join(['%s {:.4f}' % name for name in metric_names])
        metric_string = metric_format.format(*res[num_round][num]) # TODO: should be a flexible format
        print('#label: {0:d}, {1:s}'.format(k, metric_string))
    y_label.append(single_y_label)
    x_label.append(single_x_label)

res = np.array(res) # num_round x label_list_size x num_metrics
avg_res = []
std_res = []

for num, k in enumerate(args.label_list):
    # avg_res.append(np.average(res[:, num]))
    # std_res.append(np.std(res[:, num]))
    # print('#label: {0:d}, avg acc: {1:.8f}'.format(k, avg_res[-1]) + u'\u00B1{:.8f}'.format(std_res[-1]))

    avg_res.append(np.average(res[:, num, :], axis=0).tolist()) # append a list
    std_res.append(np.std(res[:, num, :], axis=0).tolist())
    metric_string_list = []
    for metric_num, name in enumerate(metric_names):
        metric_string_list.append('avg {0:s}: {1:.8f}'.format(name, avg_res[-1][metric_num]) + u'\u00B1{:.8f}'.format(std_res[-1][metric_num]))

    print('#label: {0:d}, {1:s}'.format(k, ' '.join(metric_string_list)))

# dump to file about the specific results, for ease of std computation
folder = '{}/{}/{}/'.format(args.model, args.dataset, args.method)
if not os.path.exists(folder):
    os.makedirs(folder)
prefix='knl_{:1d}slc_{:.1f}us_{:s}'.format(args.kmeans_num_layer, args.self_loop_coeff, args.uncertain_score)
for i in range(100):
    # find the next available filename
    filename = folder + prefix + '.{:02d}.json'.format(i)
    if not os.path.exists(filename):
        # parsed = {'args': vars(args), 'avg': avg_res, 'std': std_res, 'res': res.tolist()}
        parsed = {'args': vars(args), 'avg': avg_res, 'std': std_res, 'res': res.tolist(), 'x_label': x_label, 'y_label': y_label, 'time': time.time()-start_time, 'metric_names': metric_names}
        with open(filename, 'w') as f:
            f.write(json.dumps(parsed, indent=2))
        break
