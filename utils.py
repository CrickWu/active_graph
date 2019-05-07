import torch

# construct adj matrix from edge_index
# TODO: should consider GPU/CPU convertion
def convert_edge2adj(edge_index, num_edges):
    # float type
    mat = torch.zeros((num_edges, num_edges))
    for i in range(edge_index.shape[1]):
        x, y = edge_index[:, i]
        mat[x, y] = mat[y, x] = 1
    return mat

def normalize(adj):
    inv_sqrt_degree = 1. / torch.sqrt(adj.sum(dim=1, keepdim=False))
    inv_sqrt_degree[inv_sqrt_degree == float("Inf")] = 0
    return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]


