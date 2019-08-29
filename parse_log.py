import json
import os
import glob

dataset = 'Cora'
# dataset = 'Citeseer'
model = 'MatrixGCN'
# model = 'SGC'

filenames = glob.glob('./{}/{}/*/*'.format(model, dataset))

for filename in filenames:
    parsed = json.load(open(filename, 'r'))

    
    # filtering
    # print(parsed['args']['dropout'])
    # if parsed['args']['dropout'] == '0.0':
    #     print('here')

    # if parsed['args']['dropout'] == 0.5 and parsed['args']['self_loop_coeff'] == 1.0:
    if parsed['args']['dropout'] == 0.5:
        # base names
        basename = '{} {} {} {} {}'.format(parsed['args']['method'], parsed['args']['cluster_method'], parsed['args']['kmeans_num_layer'], parsed['args']['uncertain_score'], parsed['args']['self_loop_coeff'])
        print(basename, parsed['avg'])