# coding = utf-8
import numpy as np
import torch
from src.config import device
from src.DialogueSystem.build_inputs import build_graph_embed_inputs, build_manager_state_tracker_inputs, \
    build_workers_state_tracker_inputs, build_hierarchy_action_masks, build_hierarchy_manager_inputs, \
    build_workers_inputs, build_flatten_action_masks, build_flatten_manager_inputs
from src.NNModule.agents import sample_hierarchy_rl, sample_flatten_rl
from src.NNModule.utils import mask_softmax


def rollout_flatten_policy(graph_embed_field,
                           policy_field,
                           state_tracker_field,
                           state_trackers,
                           users,
                           language_generators,
                           dialogue_recorders,
                           sample_flag,
                           **models):
    # declare system mode
    if sample_flag == "random":
        mode = "RuleWarmUp"
    else:
        mode = "RL"

    # rollout
    while np.asarray([field[-1]["episode_not_end"] for field in state_tracker_field], dtype=np.bool).any():
        # forward for GNN
        graph_embed_feed_dict = build_graph_embed_inputs(graph_embed_field, state_tracker_field)
        final_node_embed = models["gnn"](graph_embed_feed_dict["initial_node_embed"],
                                         graph_embed_feed_dict["edges"],
                                         graph_embed_feed_dict["node_edges"],
                                         graph_embed_feed_dict["node_edge_mask"])

        # get the action mask
        action_masks_dict = build_flatten_action_masks(state_tracker_field, mode)

        # get the RL workers decision (batch_size, personal_node_num, 2)
        batch_size = final_node_embed.shape[0]
        rl_workers_decision = torch.zeros((batch_size, 4, 2), device=device, dtype=torch.float)

        # forward for manager state tracker
        manager_state_tracker_feed_dict = build_manager_state_tracker_inputs(graph_embed_field, final_node_embed,
                                                                             rl_workers_decision,
                                                                             state_tracker_field)
        manager_state = models["manager_state_tracker"](manager_state_tracker_feed_dict["feasible_personal_info_nodes"],
                                                        manager_state_tracker_feed_dict["workers_decision"],
                                                        manager_state_tracker_feed_dict["known_one_hot"],
                                                        manager_state_tracker_feed_dict["unknown_one_hot"],
                                                        manager_state_tracker_feed_dict["known_differ_one_hot"],
                                                        manager_state_tracker_feed_dict["total_qa_turn_one_hot"],
                                                        manager_state_tracker_feed_dict["personal_nodes"],
                                                        manager_state_tracker_feed_dict["final_node_embed"])

        # forward for manager
        manager_feed_dict = build_flatten_manager_inputs(manager_state, policy_field, final_node_embed)
        _, manager_logits = models["manager"](manager_feed_dict["manager_state"],
                                              manager_feed_dict["answer_nodes"],
                                              manager_feed_dict["graph_node_embedding"])

        # get current manager and workers policy distribution
        if mode == "RL":
            manager_probs = mask_softmax(manager_logits, action_masks_dict["rl_action_mask"], dim=1)
        else:
            manager_probs = mask_softmax(manager_logits, action_masks_dict["warm_up_action_mask"], dim=1)

        # sample from probs
        manager_action_probs = manager_probs.cpu().detach().numpy()
        answer_nodes = manager_feed_dict["answer_nodes"].cpu().detach().numpy()
        q_nodes = manager_feed_dict["q_nodes"].cpu().detach().numpy()
        sample_idxs, sample_contents = sample_flatten_rl(
            manager_action_probs,
            answer_nodes,
            q_nodes,
            sample_flag)

        # make a step
        for state_tracker, language_generator, user, dialogue_recorder, sample_idx, \
            sample_content, action_prob in zip(state_trackers,
                                               language_generators,
                                               users,
                                               dialogue_recorders,
                                               sample_idxs,
                                               sample_contents,
                                               manager_action_probs):
            state_tracker.move_a_step(language_generator,
                                      user,
                                      dialogue_recorder,
                                      sample_idx,
                                      sample_content,
                                      action_prob,
                                      mode)

    # delete the late terminal state
    for state_tracker in state_trackers:
        del state_tracker.state_tracker_field[-1]


def rollout_hierarchy_policy(graph_embed_field,
                             policy_field,
                             state_tracker_field,
                             state_trackers,
                             users,
                             language_generators,
                             dialogue_recorders,
                             sample_flag,
                             **models):
    # declare system mode
    if sample_flag == "random":
        mode = "RuleWarmUp"
    else:
        mode = "RL"

    # rollout
    while np.asarray([field[-1]["episode_not_end"] for field in state_tracker_field], dtype=np.bool).any():
        # forward for GNN
        graph_embed_feed_dict = build_graph_embed_inputs(graph_embed_field, state_tracker_field)
        final_node_embed = models["gnn"](graph_embed_feed_dict["initial_node_embed"],
                                         graph_embed_feed_dict["edges"],
                                         graph_embed_feed_dict["node_edges"],
                                         graph_embed_feed_dict["node_edge_mask"])

        # forward for workers state tracker
        workers_state_tracker_feed_dict = build_workers_state_tracker_inputs(state_tracker_field, policy_field,
                                                                             final_node_embed)
        workers_state = models["workers_state_tracker"](workers_state_tracker_feed_dict["known_one_hot"],
                                                        workers_state_tracker_feed_dict["unknown_one_hot"],
                                                        workers_state_tracker_feed_dict["known_differ_one_hot"],
                                                        workers_state_tracker_feed_dict["workers_qa_turn_one_hot"],
                                                        workers_state_tracker_feed_dict["workers_max_qa_turn_one_hot"],
                                                        workers_state_tracker_feed_dict["personal_nodes"],
                                                        workers_state_tracker_feed_dict["final_node_embed"])

        # forward for workers
        workers_feed_dict = build_workers_inputs(workers_state, policy_field, final_node_embed)
        _, workers_logits = models["workers"](workers_feed_dict["workers_state"],
                                              workers_feed_dict["answer_nodes"],
                                              workers_feed_dict["graph_node_embedding"])

        # get the action mask
        action_masks_dict = build_hierarchy_action_masks(state_tracker_field, mode)

        # get the RL workers decision (batch_size, personal_node_num, 2)
        rl_workers_probs = mask_softmax(workers_logits, action_masks_dict["rl_workers_action_mask"], dim=2).detach()
        rl_workers_decision = rl_workers_probs[:, :, -2:]

        # forward for manager state tracker
        manager_state_tracker_feed_dict = build_manager_state_tracker_inputs(graph_embed_field, final_node_embed,
                                                                             rl_workers_decision,
                                                                             state_tracker_field)
        manager_state = models["manager_state_tracker"](manager_state_tracker_feed_dict["feasible_personal_info_nodes"],
                                                        manager_state_tracker_feed_dict["workers_decision"],
                                                        manager_state_tracker_feed_dict["known_one_hot"],
                                                        manager_state_tracker_feed_dict["unknown_one_hot"],
                                                        manager_state_tracker_feed_dict["known_differ_one_hot"],
                                                        manager_state_tracker_feed_dict["total_qa_turn_one_hot"],
                                                        manager_state_tracker_feed_dict["personal_nodes"],
                                                        manager_state_tracker_feed_dict["final_node_embed"])

        # forward for manager
        manager_feed_dict = build_hierarchy_manager_inputs(manager_state, workers_state, policy_field)
        _, manager_logits = models["manager"](manager_feed_dict["manager_state"],
                                              manager_feed_dict["workers_state"])

        # get current manager and workers policy distribution
        if mode == "RL":
            manager_probs = mask_softmax(manager_logits, action_masks_dict["rl_manager_action_mask"], dim=1)
            workers_probs = mask_softmax(workers_logits, action_masks_dict["rl_workers_action_mask"], dim=2)
        else:
            manager_probs = mask_softmax(manager_logits, action_masks_dict["warm_up_manager_action_mask"], dim=1)
            workers_probs = mask_softmax(workers_logits, action_masks_dict["warm_up_workers_action_mask"], dim=2)

        # sample from probs
        manager_action_probs = manager_probs.cpu().detach().numpy()
        workers_action_probs = workers_probs.cpu().detach().numpy()
        manager_actions = manager_feed_dict["personal_nodes"].cpu().detach().numpy()
        workers_actions = workers_feed_dict["answer_nodes"].cpu().detach().numpy()
        manager_sample_idxs, manager_sample_results, workers_sample_idxs, workers_sample_results = sample_hierarchy_rl(
            manager_action_probs,
            workers_action_probs,
            manager_actions,
            workers_actions,
            sample_flag)

        # make a step
        for state_tracker, language_generator, user, dialogue_recorder, manager_sample_idx, \
            manager_sample_result, manager_action_prob, workers_sample_idx, \
            workers_sample_result, workers_action_prob in zip(state_trackers,
                                                              language_generators,
                                                              users,
                                                              dialogue_recorders,
                                                              manager_sample_idxs,
                                                              manager_sample_results,
                                                              manager_action_probs,
                                                              workers_sample_idxs,
                                                              workers_sample_results,
                                                              workers_action_probs):
            state_tracker.move_a_step(language_generator,
                                      user,
                                      dialogue_recorder,
                                      manager_sample_idx,
                                      manager_sample_result,
                                      manager_action_prob,
                                      workers_sample_idx,
                                      workers_sample_result,
                                      workers_action_prob,
                                      mode)

    # delete the late terminal state
    for state_tracker in state_trackers:
        del state_tracker.state_tracker_field[-1]
