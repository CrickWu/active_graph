import torch
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

def eval_model(model, data, test_mask):
    # return accuracy, micro_f1, macro_f1
    model.eval()
    _, pred = model(data)[1].max(dim=1)
    correct = pred[test_mask].eq(data.y[test_mask]).sum().item()
    acc = correct / test_mask.sum().item()

    # micro_f1
    model.train()
    return acc

def to_onehot(arr, n):
    # arr is one-dimensional
    # n is the maximum class number
    length = arr.shape[0]
    ret_tensor = torch.zeros(length, n, dtype=torch.float)
    indices = torch.arange(length, dtype=torch.int64) # to_device
    ret_tensor[indices, arr] = 1.
    return ret_tensor

METRIC_NAMES = ['acc', 'macro_f1'] # should be flexible and compatible with final_eval
def final_eval(model, data, test_mask, num_class):
    # return accuracy, micro_f1, macro_f1
    model.eval()
    _, pred = model(data)[1].max(dim=1)
    masked_pred = pred[test_mask]
    masked_y = data.y[test_mask]
    # accuracy
    correct = masked_pred.eq(masked_y).sum().item()
    acc = correct / test_mask.sum().item()

    # tp, fp, fn counts
    tp_list = []
    fp_list = []
    fn_list = []
    for i in range(num_class):
        tp = ((masked_pred == i) & (masked_y == i)).sum().item() # TP, dtype is int64
        fp = ((masked_pred == i) & (masked_y != i)).sum().item()
        fn = ((masked_pred != i) & (masked_y == i)).sum().item()
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
    tp_list = torch.tensor(tp_list).to(torch.float)
    fp_list = torch.tensor(fp_list).to(torch.float)
    fn_list = torch.tensor(fn_list).to(torch.float)


    # macro_f1
    prec_list = tp_list / (tp_list + fp_list)
    rec_list = tp_list / (tp_list + fn_list)
    f1_list = 2 * prec_list * rec_list / (prec_list + rec_list)
    f1_list[torch.isnan(f1_list)] = 0
    macro_f1 = f1_list.mean().item()

    model.train()

    # from sklearn.metrics import f1_score
    # # micro_f1, same as acc
    # prec = tp_list.sum() / (tp_list.sum() + fp_list.sum())
    # rec = tp_list.sum() / (tp_list.sum() + fn_list.sum())
    # micro_f1 = (2 * prec * rec / (prec + rec)).item()
    # # compare with sklearn results
    # onehot_masked_pred = to_onehot(masked_pred, num_class).numpy()
    # onehot_masked_y = to_onehot(masked_y, num_class).numpy()

    # print('pytorch, micro_f1: {}, macro_f1: {}'.format(micro_f1, macro_f1))
    # print('sklearn, micro_f1: {}, macro_f1: {}'.format(f1_score(onehot_masked_y, onehot_masked_pred, average='micro'), f1_score(onehot_masked_y, onehot_masked_pred, average='macro')))


    return acc, macro_f1
    # return acc, micro_f1, macro_f1
