import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch.nn.init import xavier_uniform_
from torch_geometric.nn import GCNConv

from utils import convert_edge2adj, normalize

def get_model(model_name):
    return eval(model_name)

class MatrixGCN(torch.nn.Module):
    def __init__(self, args, data):
        super(MatrixGCN, self).__init__()
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.mat = normalize(convert_edge2adj(data.edge_index, data.num_nodes) + torch.eye(data.num_nodes)).to(device)
        self.mat = Parameter(normalize(convert_edge2adj(data.edge_index, data.num_nodes) + torch.eye(data.num_nodes)), requires_grad=False)
        self.linear1 = Parameter(torch.Tensor(args.num_features, args.hid_dim))
        self.linear2 = Parameter(torch.Tensor(args.hid_dim, args.num_classes))

        xavier_uniform_(self.linear1)
        xavier_uniform_(self.linear2)

        self.args = args

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.mat.matmul(x.matmul(self.linear1))
        hid_x = F.relu(x)
        drop_x = F.dropout(hid_x, self.args.dropout, training=self.training)
        bef_linear2 = self.mat.matmul(drop_x)
        fin_x = bef_linear2.matmul(self.linear2)

        return (hid_x, bef_linear2, fin_x), F.log_softmax(fin_x, dim=1)

# Network definition, could be refactored
class GCN(torch.nn.Module):
    def __init__(self, args, data):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(args.num_features, args.hid_dim)
        self.conv2 = GCNConv(args.hid_dim, args.num_classes)
        self.args = args

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        hid_x = F.relu(x)
        x = F.dropout(hid_x, self.args.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        # TODO: the final element in the triple is added for compatability
        return (hid_x, x, x), F.log_softmax(x, dim=1)
    
class SGC(torch.nn.Module):
    def __init__(self, args, data):
        super(SGC, self).__init__()
        self.mat = Parameter(normalize(convert_edge2adj(data.edge_index, data.num_nodes) + torch.eye(data.num_nodes)), requires_grad=False)
        self.linear1 = Parameter(torch.Tensor(args.num_features, args.hid_dim))
        self.linear2 = Parameter(torch.Tensor(args.hid_dim, args.num_classes))

        xavier_uniform_(self.linear1)
        xavier_uniform_(self.linear2)

        self.args = args

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        hid_x = self.mat.matmul(x.matmul(self.linear1))
        drop_x = F.dropout(hid_x, self.args.dropout, training=self.training)
        bef_linear2 = self.mat.matmul(drop_x)
        fin_x = bef_linear2.matmul(self.linear2)

        return (hid_x, bef_linear2, fin_x), F.log_softmax(fin_x, dim=1)