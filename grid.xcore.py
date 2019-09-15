import subprocess, os
import tqdm
my_env = os.environ.copy()
my_env['CUDA_VISIBLE_DEVICES'] = '3'
# my_env['CUDA_VISIBLE_DEVICES'] = '2'

#
ori_args = 'python active_graph.py --lr 0.01 --label_list 5 10 20 40 80 160 --epoch 200 --rand_rounds 5'.split()

# args = ['python', 'active_graph.py', '--method', 'uncertain', '--lr', '0.01', '--label_list', '5 10 20 40 80', '--epoch', '200', '--rand_rounds', '1', '--dataset', 'Cora', '--model', 'GCN', '--uncertain_score' 'entropy']

def run(args, my_env):
    print(args)
    p = subprocess.Popen(args, env=my_env)
    (output, err) = p.communicate()  
    #This makes the wait possible
    p_status = p.wait()


# models = ['MatrixGCN', 'SGC', 'GCN']
models = ['MatrixGCN']
# datasets = ['PPI{}'.format(i) for i in range(0, 5)]
# datasets = ['Cora', 'Citeseer', 'PubMed']
datasets = ['Cora', 'Citeseer']
# datasets = ['Computers']
# datasets = ['CoraFull', 'Photos']
# kmeans
for model in models:
    for dataset in datasets:
        model_args = ['--model', model]
        dataset_args = ['--dataset', dataset]
        # for dropout in ['0.5', '0.']:
        for dropout in ['0.5']:

            extra_args = model_args + dataset_args + ['--dropout', dropout]

            # xcoresetmip
            # for cluster_method in ['kcenter']:
            coreset_args = ['--method', 'xcoresetmip']
                # run
            args = ori_args + coreset_args + extra_args
            run(args, my_env)