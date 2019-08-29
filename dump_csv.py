import json
import os
import glob
import csv

# dataset = 'Cora'
# # dataset = 'Citeseer'
# model = 'MatrixGCN'
# # model = 'SGC'

# with open('computers.csv', 'w') as csvfile:
with open('test.csv', 'w') as csvfile:
    header = ['model', 'dataset', 'method', 'cluster_method', 
    'kmeans_num_layer', 'uncertain_score', 
    'self_loop_coeff', 'dropout', 'time']
    writer = csv.writer(csvfile)
    writer.writerow(header + [5, 10, 20, 40, 80] + [5, 10, 20, 40, 80] + ['filename'])
    # for model in ['MatrixGCN', 'SGC', 'GCN']:
    for model in ['MatrixGCN', 'SGC']:
        for dataset in ['Cora', 'Citeseer', 'PubMed', 'CoraFull']:
            filenames = glob.glob('./{}/{}/*/*'.format(model, dataset))
            for filename in filenames:
                parsed = json.load(open(filename, 'r'))
                # Only keep results from 2019.5.21
                if 'time' not in parsed:
                    continue
                if parsed['args']['rand_rounds'] < 5:
                    continue
                args = [model, dataset, parsed['args']['method'], 
                parsed['args']['cluster_method'], 
                parsed['args']['kmeans_num_layer'], parsed['args']['uncertain_score'], 
                parsed['args']['self_loop_coeff'], parsed['args']['dropout'], parsed['time']]
                row = args + parsed['avg'] + parsed['std'] + [filename]
                writer.writerow(row)