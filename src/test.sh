#!/usr/bin/env bash
# Test Rule based systems
CUDA_VISIBLE_DEVICES=0 nohup python main.py --record_dialogue --model_setting hrl --test --test_rule_based_system \
    --use_hierarchy_policy > test_hierarchy_rule_based_system.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python main.py --model_setting rl --test --test_rule_based_system \
    > test_flatten_rule_based_system.log 2>&1 &

# Test data-driven systems
CUDA_VISIBLE_DEVICES=0 nohup python main.py --record_dialogue --model_setting ghrl --test \
    --use_hierarchy_policy --use_graph_based_state_tracker --new_node_emb_size_list 40 50 \
    --trained_model RL/epoch_300_success_0.8921875_turn_9.10390625 > test_ghrl.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py --model_setting grl --test \
    --use_graph_based_state_tracker --new_node_emb_size_list 40 50 \
    --trained_model RL/epoch_63_success_0.58125_turn_8.0 > test_grl.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python main.py --record_dialogue --model_setting hrl --test \
    --use_hierarchy_policy \
    --trained_model RL/epoch_288_success_0.83984375_turn_7.69765625 > test_hrl.log 2>&1 &

