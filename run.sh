#!/bin/bash

# for a*2 kmeans
# python active_graph.py --method kmeans --lr 0.01 --label_list 10 --epoch 200 --rand_rounds 5 --kmeans_num_layer=2 --dataset Cora

# for normal kmeans
# python active_graph.py --lr 0.01 --method kmeans "$@" # appending parameters


# CUDA_VISIBLE_DEVICES=3 python active_graph.py --method kmeans --lr 0.01 --label_list 10 20 40 80 --epoch 200 --rand_rounds 5 --kmeans_num_layer 2 --dataset Citeseer --self_loop_coeff 1.

# CUDA_VISIBLE_DEVICES=3 python active_graph.py --method kmeans --lr 0.01 --label_list 10 20 40 80 --epoch 200 --rand_rounds 5 --kmeans_num_layer 2 --dataset Citeseer --self_loop_coeff 0. --model GCN

# CUDA_VISIBLE_DEVICES=3 python active_graph.py --method kmeans --lr 0.01 --label_list 10 20 40 80 --epoch 200 --rand_rounds 5 --kmeans_num_layer 2 --dataset Citeseer --self_loop_coeff 1. --model MatrixGCN

# CUDA_VISIBLE_DEVICES=3 python active_graph.py --method kmeans --lr 0.01 --label_list 10 20 40 80 --epoch 200 --rand_rounds 5 --kmeans_num_layer 2 --dataset Citeseer --self_loop_coeff 0. --model MatrixGCN

# CUDA_VISIBLE_DEVICES=3 python active_graph.py --method kmeans --lr 0.01 --label_list 10 20 40 80 --epoch 200 --rand_rounds 5 --kmeans_num_layer 2 --dataset Cora --self_loop_coeff 1.

# CUDA_VISIBLE_DEVICES=3 python active_graph.py --method kmeans --lr 0.01 --label_list 10 20 40 80 --epoch 200 --rand_rounds 5 --kmeans_num_layer 2 --dataset Cora --self_loop_coeff 0. --model GCN

# CUDA_VISIBLE_DEVICES=3 python active_graph.py --method kmeans --lr 0.01 --label_list 10 20 40 80 --epoch 200 --rand_rounds 5 --kmeans_num_layer 2 --dataset Cora --self_loop_coeff 1. --model MatrixGCN

# CUDA_VISIBLE_DEVICES=3 python active_graph.py --method kmeans --lr 0.01 --label_list 10 20 40 80 --epoch 200 --rand_rounds 5 --kmeans_num_layer 2 --dataset Cora --self_loop_coeff 0. --model MatrixGCN

CUDA_VISIBLE_DEVICES=2 python active_graph.py --method uncertain --lr 0.01 --label_list 5 10 20 40 80 --epoch 200 --rand_rounds 5 --dataset Cora --model GCN --uncertain_score entropy

CUDA_VISIBLE_DEVICES=2 python active_graph.py --method coreset --lr 0.01 --label_list 5 10 20 40 80 --epoch 200 --rand_rounds 1 --dataset Cora --model GCN --cluster_method kcenter

CUDA_VISIBLE_DEVICES=2 python active_graph.py --method kmeans --lr 0.01 --label_list 10 20 40 80 --epoch 200 --rand_rounds 1 --kmeans_num_layer 2 --dataset Cora --self_loop_coeff 0. --model GCN --cluster_method kcenter