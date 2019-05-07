#!/bin/bash

# for a*2 kmeans
# python active_graph.py --method kmeans --lr 0.01 --label_list 10 --epoch 200 --rand_rounds 5 --kmeans_num_layer=2 --dataset Cora

# for normal kmeans
python active_graph.py --lr 0.01 --method kmeans "$@" # appending parameters
