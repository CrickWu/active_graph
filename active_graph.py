import numpy as np
import argparse
import os
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid, PPI

from methods import ActiveFactory
from models import get_model

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

    print(pred.dtype, data.y.dtype)
    correct = pred[test_mask].eq(data.y[test_mask]).sum().item()
    acc = correct / test_mask.sum().item()
    model.train()
    return acc

def eval_model_f1(model, data, data_y, test_mask):
    model.eval()
    # TODO: whehther transform it into int?
    pred = model(data)[0][2] > 0.5
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
torch.manual_seed(args.seed)  # for GPU and CPU after torch 1.0
np.random.seed(args.seed)

# device specification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.dataset[:3] == 'PPI':
    args.multilabel = True
    dataset = PPI(root='./data/PPI')
    dataset_num = int(args.dataset[3:])
    data = dataset[dataset_num].to(device)
else:
    dataset = Planetoid(root='./data/{}'.format(args.dataset), name='{}'.format(args.dataset))
    data = dataset[0].to(device)
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
        if args.multilabel:
            acc = eval_model_f1(model, data, data_y, test_mask)
        else:
            acc = eval_model(model, data, data_y, test_mask)
        if args.verbose:
            print('epoch {} acc: {:.4f} loss: {:.4f}'.format(epoch, acc, loss.item()))
    return acc, train_mask, model, optimizer


res = np.zeros((args.rand_rounds, len(args.label_list)))
print('Using', device, 'for neural network training')
# different random seeds
for num_round in range(args.rand_rounds):
    train_mask = None
    model = Net(args, data)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for num, k in enumerate(args.label_list):
        # lr should be 0.001??
        # replace old model, optimizer with new model
        acc, train_mask, model, optimizer = active_learn(k, data, model, optimizer, train_mask, args)

        res[num_round, num] = acc
        print('#label: {0:d}, acc: {1:.4f}'.format(k, res[num_round, num]))

avg_res = []
std_res = []

for num, k in enumerate(args.label_list):
    avg_res.append(np.average(res[:, num]))
    std_res.append(np.std(res[:, num]))
    print('#label: {0:d}, avg acc: {1:.8f}'.format(k, avg_res[-1]) + u'\u00B1{:.8f}'.format(std_res[-1]))

# dump to file about the specific results, for ease of std computation
folder = '{}/{}/{}/'.format(args.model, args.dataset, args.method)
if not os.path.exists(folder):
    os.makedirs(folder)
# find next available filvars(
prefix='knl_{:1d}slc_{:.1f}us_{:s}'.format(args.kmeans_num_layer, args.self_loop_coeff, args.uncertain_score)
for i in range(100):
    filename = folder + prefix + '.{:02d}.json'.format(i)
    if not os.path.exists(filename):
        parsed = {'args': vars(args), 'avg': avg_res, 'std': std_res, 'res': res.tolist()}
        with open(filename, 'w') as f:
            f.write(json.dumps(parsed, indent=2))
        break
