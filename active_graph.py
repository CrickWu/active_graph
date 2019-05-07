import numpy as np
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

import torch_geometric.utils as utils
from torch_geometric.datasets import Planetoid, CoraFull
from torch_geometric.nn import GCNConv

from methods import ActiveFactory

# Network definition, could be refactored
class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = GCNConv(args.num_features, args.hid_dim)
        self.conv2 = GCNConv(args.hid_dim, args.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x, F.log_softmax(x, dim=1)

# Tool functions
def eval_model(model, data, test_mask):
    model.eval()
    _, pred = model(data)[1].max(dim=1)
    correct = pred[test_mask].eq(data.y[test_mask]).sum().item()
    acc = correct / test_mask.sum().item()
    model.train()
    return acc


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
parser.add_argument('--method', type=str, default='random',
                    help='choice between [random, kmeans, a2kmeans]')
parser.add_argument('--rand_rounds', type=int, default=1,
                    help='number of rounds for averaging scores')

# KMeans
parser.add_argument('--kmeans_num_layer', type=int, default=0,
                    help='number of propergation for KMeans to generate new features')

# dataset parsed infor
parser.add_argument('--num_features', type=int, default=None,
                    help='initial feature dimension for input dataset')
parser.add_argument('--num_classes', type=int, default=None,
                    help='number of classes for node classification')

# TODO: replace with the pseudo-command line
args = parser.parse_args()

# device specification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='./data/{}'.format(args.dataset), name='{}'.format(args.dataset))

# preprocessing of data and model
torch.manual_seed(args.seed)  # for GPU and CPU after torch 1.0
np.random.seed(args.seed)

data = dataset[0].to(device)
args.num_features = dataset.num_features 
args.num_classes = dataset.num_classes

print(args)

# 2 types of AL
# - 1. fresh start of optimizer and model
# - 2. fresh start of optimizer and NOT model

# TODO: should consider interactive selection of nodes
def active_learn(k, data, old_model, old_optimizer, prev_index, args):
    test_mask = torch.ones_like(data.test_mask)

    learner = ActiveFactory(args, old_model, data, prev_index).get_learner()
    train_mask = learner.pretrain_choose(k)

    model = Net(args)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    
    if args.verbose:
        print('selected labels:', data.y[train_mask])
    # fresh new training
    for epoch in tqdm(range(args.epoch)):
        # Optimize GCN
        optimizer.zero_grad()
        pre_out, out = model(data)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        acc = eval_model(model, data, test_mask)
        if args.verbose:
            print('epoch {} acc: {:.4f} loss: {:.4f}'.format(epoch, acc, loss.item()))
    return acc, train_mask, model, optimizer


res = np.zeros((args.rand_rounds, len(args.label_list)))
print('Using', device, 'for neural network training')
# different random seeds
for num_round in range(args.rand_rounds):
    train_mask = None
    model = Net(args)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for num, k in enumerate(args.label_list):
        # lr should be 0.001??
        # replace old model, optimizer with new model
        acc, train_mask, model, optimizer = active_learn(k, data, model, optimizer, train_mask, args)

        res[num_round, num] = acc
        print('#label: {0:d}, acc: {1:.4f}'.format(k, res[num_round, num]))

for num, k in enumerate(args.label_list):
    print('#label: {0:d}, avg acc: {1:.8f}'.format(k, np.average(res[:, num])))
