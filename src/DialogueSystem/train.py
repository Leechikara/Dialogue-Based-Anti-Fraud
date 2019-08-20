# coding = utf-8
import torch
import torch.nn as nn
from src.DialogueSystem.build_loss import build_hierarchy_loss, build_flatten_loss
from src.DialogueSystem.build_inputs import build_graph_embed_inputs, build_manager_state_tracker_inputs, \
    build_workers_state_tracker_inputs, build_hierarchy_action_masks, build_hierarchy_manager_inputs, \
    build_workers_inputs, build_flatten_manager_inputs
from src.NNModule.utils import mask_softmax
from src.config import device


def dissolve_time_step(*args, **kwargs):
    """ for tensor with shape (batch_size, time_step, *), we reshape to (batch_size * time_step, *) """
    dissolved_args = list()
    for arg in args:
        batch_size, time_step = arg.shape[0], arg.shape[1]
        dissolved_args.append(arg.reshape(batch_size * time_step, *arg.shape[2:]))

    dissolved_kwargs = dict()
    for key, value in kwargs.items():
        batch_size, time_step = value.shape[0], value.shape[1]
        dissolved_kwargs[key] = value.reshape(batch_size * time_step, *value.shape[2:])

    return batch_size, time_step, tuple(dissolved_args), dissolved_kwargs


def separate_time_step(batch_size, time_step, *args, **kwargs):
    """  the inverse operation of dissolve_time_step  """
    separated_args = list()
    for arg in args:
        separated_args.append(arg.reshape(batch_size, time_step, *arg.shape[1:]))

    separated_kwargs = dict()
    for key, value in kwargs.items():
        separated_kwargs[key] = value.reshape(batch_size, time_step, *value.shape[1:])
    return tuple(separated_args), separated_kwargs


def train_flatten_policy(graph_embed_field,
                         policy_field,
                         state_tracker_field,
                         optimizer,
                         max_clip,
                         entropy_coef,
                         mode,
                         **models):
    # forward for GNN
    graph_embed_feed_dict = build_graph_embed_inputs(graph_embed_field, state_tracker_field, rollout=False)
    batch_size, time_step, _, graph_embed_feed_dict = dissolve_time_step(**graph_embed_feed_dict)
    final_node_embed = models["gnn"](graph_embed_feed_dict["initial_node_embed"],
                                     graph_embed_feed_dict["edges"],
                                     graph_embed_feed_dict["node_edges"],
                                     graph_embed_feed_dict["node_edge_mask"])
    (final_node_embed,), _ = separate_time_step(batch_size, time_step, final_node_embed)

    # get the RL workers decision (batch_size, time_step, personal_node_num, 2)
    rl_workers_decision = torch.zeros((batch_size, time_step, 4, 2), device=device, dtype=torch.float)

    # forward for manager state tracker
    manager_state_tracker_feed_dict = build_manager_state_tracker_inputs(graph_embed_field, final_node_embed,
                                                                         rl_workers_decision, state_tracker_field,
                                                                         rollout=False)
    batch_size, time_step, _, manager_state_tracker_feed_dict = dissolve_time_step(**manager_state_tracker_feed_dict)
    manager_state = models["manager_state_tracker"](manager_state_tracker_feed_dict["feasible_personal_info_nodes"],
                                                    manager_state_tracker_feed_dict["workers_decision"],
                                                    manager_state_tracker_feed_dict["known_one_hot"],
                                                    manager_state_tracker_feed_dict["unknown_one_hot"],
                                                    manager_state_tracker_feed_dict["known_differ_one_hot"],
                                                    manager_state_tracker_feed_dict["total_qa_turn_one_hot"],
                                                    manager_state_tracker_feed_dict["personal_nodes"],
                                                    manager_state_tracker_feed_dict["final_node_embed"])
    (manager_state,), _ = separate_time_step(batch_size, time_step, manager_state)

    # forward for manager
    manager_feed_dict = build_flatten_manager_inputs(manager_state, policy_field, final_node_embed, rollout=False)
    batch_size, time_step, _, manager_feed_dict = dissolve_time_step(**manager_feed_dict)
    manager_values, manager_logits = models["manager"](manager_feed_dict["manager_state"],
                                                       manager_feed_dict["answer_nodes"],
                                                       manager_feed_dict["graph_node_embedding"])
    (manager_values, manager_logits), _ = separate_time_step(batch_size, time_step,
                                                             manager_values,
                                                             manager_logits)

    # build loss
    policy_loss, value_loss, entropy = build_flatten_loss(state_tracker_field, manager_logits, manager_values, mode)
    loss = policy_loss + value_loss + entropy * entropy_coef

    optimizer_step(optimizer, loss, max_clip, **models)

    return loss.item()


def train_hierarchy_policy(graph_embed_field,
                           policy_field,
                           state_tracker_field,
                           optimizer,
                           max_clip,
                           entropy_coef,
                           mode,
                           **models):
    # forward for GNN
    graph_embed_feed_dict = build_graph_embed_inputs(graph_embed_field, state_tracker_field, rollout=False)
    batch_size, time_step, _, graph_embed_feed_dict = dissolve_time_step(**graph_embed_feed_dict)
    final_node_embed = models["gnn"](graph_embed_feed_dict["initial_node_embed"],
                                     graph_embed_feed_dict["edges"],
                                     graph_embed_feed_dict["node_edges"],
                                     graph_embed_feed_dict["node_edge_mask"])
    (final_node_embed,), _ = separate_time_step(batch_size, time_step, final_node_embed)

    # forward for workers state tracker
    workers_state_tracker_feed_dict = build_workers_state_tracker_inputs(state_tracker_field, policy_field,
                                                                         final_node_embed, rollout=False)
    batch_size, time_step, _, workers_state_tracker_feed_dict = dissolve_time_step(**workers_state_tracker_feed_dict)
    workers_state = models["workers_state_tracker"](workers_state_tracker_feed_dict["known_one_hot"],
                                                    workers_state_tracker_feed_dict["unknown_one_hot"],
                                                    workers_state_tracker_feed_dict["known_differ_one_hot"],
                                                    workers_state_tracker_feed_dict["workers_qa_turn_one_hot"],
                                                    workers_state_tracker_feed_dict["workers_max_qa_turn_one_hot"],
                                                    workers_state_tracker_feed_dict["personal_nodes"],
                                                    workers_state_tracker_feed_dict["final_node_embed"])
    (workers_state,), _ = separate_time_step(batch_size, time_step, workers_state)

    # forward for workers
    workers_feed_dict = build_workers_inputs(workers_state, policy_field, final_node_embed, rollout=False)
    batch_size, time_step, _, workers_feed_dict = dissolve_time_step(**workers_feed_dict)
    workers_values, workers_logits = models["workers"](workers_feed_dict["workers_state"],
                                                       workers_feed_dict["answer_nodes"],
                                                       workers_feed_dict["graph_node_embedding"])
    (workers_values, workers_logits), _ = separate_time_step(batch_size, time_step, workers_values, workers_logits)

    # get the action mask
    action_masks_dict = build_hierarchy_action_masks(state_tracker_field, mode, rollout=False)

    # get the RL workers decision (batch_size, time_step, personal_node_num, 2)
    rl_workers_probs = mask_softmax(workers_logits, action_masks_dict["rl_workers_action_mask"], dim=3).detach()
    rl_workers_decision = rl_workers_probs[:, :, :, -2:]

    # forward for manager state tracker
    manager_state_tracker_feed_dict = build_manager_state_tracker_inputs(graph_embed_field, final_node_embed,
                                                                         rl_workers_decision, state_tracker_field,
                                                                         rollout=False)
    batch_size, time_step, _, manager_state_tracker_feed_dict = dissolve_time_step(**manager_state_tracker_feed_dict)
    manager_state = models["manager_state_tracker"](manager_state_tracker_feed_dict["feasible_personal_info_nodes"],
                                                    manager_state_tracker_feed_dict["workers_decision"],
                                                    manager_state_tracker_feed_dict["known_one_hot"],
                                                    manager_state_tracker_feed_dict["unknown_one_hot"],
                                                    manager_state_tracker_feed_dict["known_differ_one_hot"],
                                                    manager_state_tracker_feed_dict["total_qa_turn_one_hot"],
                                                    manager_state_tracker_feed_dict["personal_nodes"],
                                                    manager_state_tracker_feed_dict["final_node_embed"])
    (manager_state,), _ = separate_time_step(batch_size, time_step, manager_state)

    # forward for manager
    manager_feed_dict = build_hierarchy_manager_inputs(manager_state, workers_state, policy_field, rollout=False)
    batch_size, time_step, _, manager_feed_dict = dissolve_time_step(**manager_feed_dict)
    manager_values, manager_logits = models["manager"](manager_feed_dict["manager_state"],
                                                       manager_feed_dict["workers_state"])
    (manager_values, manager_logits), _ = separate_time_step(batch_size, time_step,
                                                             manager_values,
                                                             manager_logits)

    # build loss
    manager_policy_loss, workers_policy_loss, \
        manager_value_loss, workers_value_loss, \
        manager_entropy, workers_entropy = build_hierarchy_loss(state_tracker_field,
                                                                manager_logits,
                                                                manager_values,
                                                                workers_logits,
                                                                workers_values,
                                                                mode)
    loss = manager_policy_loss + workers_policy_loss + \
        manager_value_loss + workers_value_loss + \
        (manager_entropy + workers_entropy) * entropy_coef

    optimizer_step(optimizer, loss, max_clip, **models)

    return loss.item()


def optimizer_step(optimizer, loss, max_clip, **models):
    optimizer.zero_grad()
    loss.backward()
    params = list()
    for model in models.values():
        params.extend(model.parameters())
    nn.utils.clip_grad_norm_(params, max_clip)
    optimizer.step()
