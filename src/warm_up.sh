#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 nohup python main.py --model_setting ghrl --train --warm_up \
    --use_hierarchy_policy --use_graph_based_state_tracker --new_node_emb_size_list 40 50 > warm_up_ghrl.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py --model_setting grl --train --warm_up \
    --use_graph_based_state_tracker --new_node_emb_size_list 40 50 > warm_up_grl.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python main.py --model_setting hrl --train --warm_up \
    --use_hierarchy_policy > warm_up_hrl.log 2>&1 &