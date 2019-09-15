import json
import os
import glob
import csv

# dataset = 'Cora'
# # dataset = 'Citeseer'
# model = 'MatrixGCN'
# # model = 'SGC'

# with open('computers.csv', 'w') as csvfile:
with open('main.csv', 'w') as csvfile:
    header = ['model', 'dataset', 'method', 'cluster_method', 
    'kmeans_num_layer', 'uncertain_score', 
    'self_loop_coeff', 'dropout', 'time']
    writer = csv.writer(csvfile)
    writer.writerow(header + [5, 10, 20, 40, 80, 160] + [5, 10, 20, 40, 80, 160] + ['metric_names', 'filename'])
    # for model in ['MatrixGCN', 'SGC', 'GCN']:
    for model in ['MatrixGCN', 'SGC']:
        for dataset in ['Cora', 'Citeseer', 'PubMed', 'CoraFull', 'Photo']:
            filenames = glob.glob('./{}/{}/*/*'.format(model, dataset))
            for filename in filenames:
                parsed = json.load(open(filename, 'r'))

                args = [model, dataset, parsed['args']['method'], 
                parsed['args']['cluster_method'], 
                parsed['args']['kmeans_num_layer'], parsed['args']['uncertain_score'], 
                parsed['args']['self_loop_coeff'], parsed['args']['dropout'], parsed['time']]
                row = args + parsed['avg'] + parsed['std'] + [parsed['metric_names']] + [filename]
                writer.writerow(row)