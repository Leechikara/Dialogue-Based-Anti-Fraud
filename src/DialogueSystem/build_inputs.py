# coding = utf-8
import torch
import numpy as np
from src.config import device, MaxWorkerExploringTimeStep, MaxExploringTimeStep


def build_graph_embed_inputs(graph_embed_field, state_tracker_field, rollout=True):
    """
    :param graph_embed_field:
    :param state_tracker_field:
    :param rollout: if rollout is False, * is the time_step.
    :return:
    initial_node_embed: (batch_size, *, max_node_num, feature_size)
    edges: (batch_size, *, max_edge_num, 3)
    node_edges: (batch_size, *, max_node_num, max_node_edge_num)
    node_edge_mask: (batch_size, *, max_node_num, max_node_edge_num)
    """
    edges = graph_embed_field["edges"]
    node_edges = graph_embed_field["node_edges"]
    node_edge_mask = graph_embed_field["node_edge_mask"]
    static_feature = graph_embed_field["static_feature"]

    if rollout is True:
        dialogue_feature = np.concatenate([[field[-1]["dialogue_feature"]] for field in state_tracker_field])
    else:
        time_step = len(state_tracker_field[0])
        dialogue_feature = np.concatenate(
            [[np.concatenate([[state["dialogue_feature"]] for state in field])] for field in
             state_tracker_field])
        edges = np.repeat(edges[:, np.newaxis], time_step, axis=1)
        node_edges = np.repeat(node_edges[:, np.newaxis], time_step, axis=1)
        node_edge_mask = np.repeat(node_edge_mask[:, np.newaxis], time_step, axis=1)
        static_feature = np.repeat(static_feature[:, np.newaxis], time_step, axis=1)

    initial_node_embed = np.concatenate([static_feature, dialogue_feature], axis=-1)
    feed_dict = dict()
    feed_dict["initial_node_embed"] = torch.Tensor(initial_node_embed).to(device=device, dtype=torch.float)
    feed_dict["edges"] = torch.Tensor(edges).to(device=device, dtype=torch.long)
    feed_dict["node_edges"] = torch.Tensor(node_edges).to(device=device, dtype=torch.long)
    feed_dict["node_edge_mask"] = torch.Tensor(node_edge_mask).to(device=device, dtype=torch.uint8)
    return feed_dict


def to_one_hot(vec, size):
    """
    :param vec: any shape vector
    :param size: the one hot size
    :return: one hot vector of tensor
    """
    shape = vec.shape
    vec_flat = vec.reshape(-1)

    one_hot_vec = np.zeros((vec_flat.shape[0], size))
    one_hot_vec[np.arange(vec_flat.shape[0]), vec_flat] = 1
    one_hot_vec = one_hot_vec.reshape((*shape, size))
    return one_hot_vec


def build_workers_state_tracker_inputs(state_tracker_field, policy_field, final_node_embed, rollout=True):
    """

    :param state_tracker_field:
    :param policy_field:
    :param final_node_embed: (batch_size, *, max_node_num, feature_size)
    :param rollout: if rollout is False, * is the time_step.
    :return:
    known_one_hot: (batch_size, *, workers_num, feature_size)
    unknown_one_hot: (batch_size, *, workers_num, feature_size)
    known_differ_one_hot: (batch_size, *, workers_num, feature_size)
    workers_qa_turn_one_hot: (batch_size, *, workers_num, feature_size)
    workers_max_qa_turn_one_hot: (batch_size, *, workers_num, feature_size)
    personal_nodes: (batch_size, *, personal_node_num)
    final_node_embed: (batch_size, *, feature_size)
    """
    personal_nodes = policy_field["manager_actions"]
    workers_max_qa_turn = policy_field["workers_max_qa_turn"]

    if rollout is True:
        known = np.concatenate(
            [[np.concatenate([[item["Known"]] for item in field[-1]["workers_counter"]])] for field in
             state_tracker_field])
        unknown = np.concatenate(
            [[np.concatenate([[item["UnKnown"]] for item in field[-1]["workers_counter"]])] for field in
             state_tracker_field])
        known_differ = np.concatenate(
            [[np.concatenate(
                [[(item["Known"] - item["UnKnown"]) if (item["Known"] - item["UnKnown"]) > 0 else 0] for item in
                 field[-1]["workers_counter"]])] for field in state_tracker_field])
        workers_qa_turn = np.concatenate([[field[-1]["workers_qa_turn"]] for field in state_tracker_field])
    else:
        time_step = final_node_embed.shape[1]
        personal_nodes = np.repeat(personal_nodes[:, np.newaxis], time_step, axis=1)
        workers_max_qa_turn = np.repeat(workers_max_qa_turn[:, np.newaxis], time_step, axis=1)

        known = np.concatenate([[np.concatenate(
            [[np.concatenate([[item["Known"]] for item in state["workers_counter"]])] for state in field])] for field in
            state_tracker_field])
        unknown = np.concatenate([[np.concatenate(
            [[np.concatenate([[item["UnKnown"]] for item in state["workers_counter"]])] for state in field])] for field
            in state_tracker_field])
        known_differ = np.concatenate([[np.concatenate([[np.concatenate(
            [[(item["Known"] - item["UnKnown"]) if (item["Known"] - item["UnKnown"]) > 0 else 0] for item in
             state["workers_counter"]])] for state in field])] for field in state_tracker_field])
        workers_qa_turn = np.concatenate(
            [[np.concatenate([[state["workers_qa_turn"]] for state in field])] for field in state_tracker_field])

    known_one_hot = to_one_hot(known, size=(MaxWorkerExploringTimeStep + 1))
    unknown_one_hot = to_one_hot(unknown, size=(MaxWorkerExploringTimeStep + 1))
    known_differ_one_hot = to_one_hot(known_differ, size=(MaxWorkerExploringTimeStep + 1))
    workers_qa_turn_one_hot = to_one_hot(workers_qa_turn, size=(MaxWorkerExploringTimeStep + 1))
    workers_max_qa_turn_one_hot = to_one_hot(workers_max_qa_turn, size=(MaxWorkerExploringTimeStep + 1))

    feed_dict = dict()
    feed_dict["known_one_hot"] = torch.Tensor(known_one_hot).to(device=device, dtype=torch.float)
    feed_dict["unknown_one_hot"] = torch.Tensor(unknown_one_hot).to(device=device, dtype=torch.float)
    feed_dict["known_differ_one_hot"] = torch.Tensor(known_differ_one_hot).to(device, dtype=torch.float)
    feed_dict["workers_qa_turn_one_hot"] = torch.Tensor(workers_qa_turn_one_hot).to(device, dtype=torch.float)
    feed_dict["workers_max_qa_turn_one_hot"] = torch.Tensor(workers_max_qa_turn_one_hot).to(device, dtype=torch.float)
    feed_dict["personal_nodes"] = torch.Tensor(personal_nodes).to(device=device, dtype=torch.long)
    feed_dict["final_node_embed"] = final_node_embed
    return feed_dict


def build_manager_state_tracker_inputs(graph_embed_field, final_node_embed, workers_decision, state_tracker_field,
                                       rollout=True):
    """
    :param graph_embed_field:
    :param final_node_embed: (batch_size, *, max_node_num, feature_size)
    :param workers_decision: (batch_size, *, workers_num, 2)
    :param state_tracker_field:
    :param rollout: if rollout is False, * is the time_step.
    :return:
    feasible_personal_info_nodes: (batch_size, *, workers_num)
    workers_decision: (batch_size, *, workers_num, 2)
    known_one_hot: (batch_size, *, workers_num, feature_size)
    unknown_one_hot: (batch_size, *, workers_num, feature_size)
    known_differ_one_hot: (batch_size, *, workers_num, feature_size)
    total_qa_turn_one_hot: (batch_size, *, feature_size)
    personal_nodes: (batch_size, *, workers_num)
    final_node_embed: (batch_size, *, max_node_num, feature_size)
    """
    feasible_personal_info_nodes = graph_embed_field["feasible_personal_info_nodes"]
    personal_nodes = graph_embed_field["personal_nodes"]

    if rollout is True:
        known = np.concatenate(
            [[np.concatenate([[item["Known"]] for item in field[-1]["workers_counter"]])] for field in
             state_tracker_field])
        unknown = np.concatenate(
            [[np.concatenate([[item["UnKnown"]] for item in field[-1]["workers_counter"]])] for field in
             state_tracker_field])
        known_differ = np.concatenate(
            [[np.concatenate(
                [[(item["Known"] - item["UnKnown"]) if (item["Known"] - item["UnKnown"]) > 0 else 0] for item in
                 field[-1]["workers_counter"]])] for field in state_tracker_field])
        total_qa_turn = np.concatenate([[field[-1]["total_qa_turn"]] for field in state_tracker_field])
    else:
        time_step = final_node_embed.shape[1]
        feasible_personal_info_nodes = np.repeat(feasible_personal_info_nodes[:, np.newaxis], time_step, axis=1)
        personal_nodes = np.repeat(personal_nodes[:, np.newaxis], time_step, axis=1)

        known = np.concatenate([[np.concatenate(
            [[np.concatenate([[item["Known"]] for item in state["workers_counter"]])] for state in field])] for field in
            state_tracker_field])
        unknown = np.concatenate([[np.concatenate(
            [[np.concatenate([[item["UnKnown"]] for item in state["workers_counter"]])] for state in field])] for field
            in
            state_tracker_field])
        known_differ = np.concatenate([[np.concatenate([[np.concatenate(
            [[(item["Known"] - item["UnKnown"]) if (item["Known"] - item["UnKnown"]) > 0 else 0] for item in
             state["workers_counter"]])] for state in field])] for field in state_tracker_field])
        total_qa_turn = np.concatenate(
            [[np.concatenate([[state["total_qa_turn"]] for state in field])] for field in state_tracker_field])

    known_one_hot = to_one_hot(known, size=(MaxWorkerExploringTimeStep + 1))
    unknown_one_hot = to_one_hot(unknown, size=(MaxWorkerExploringTimeStep + 1))
    known_differ_one_hot = to_one_hot(known_differ, size=(MaxWorkerExploringTimeStep + 1))
    total_qa_turn_one_hot = to_one_hot(total_qa_turn, size=(MaxExploringTimeStep + 1))

    feed_dict = dict()
    feed_dict["feasible_personal_info_nodes"] = torch.Tensor(feasible_personal_info_nodes).to(device=device,
                                                                                              dtype=torch.float)
    feed_dict["workers_decision"] = workers_decision
    feed_dict["known_one_hot"] = torch.Tensor(known_one_hot).to(device=device, dtype=torch.float)
    feed_dict["unknown_one_hot"] = torch.Tensor(unknown_one_hot).to(device=device, dtype=torch.float)
    feed_dict["known_differ_one_hot"] = torch.Tensor(known_differ_one_hot).to(device, dtype=torch.float)
    feed_dict["total_qa_turn_one_hot"] = torch.Tensor(total_qa_turn_one_hot).to(device, dtype=torch.float)
    feed_dict["personal_nodes"] = torch.Tensor(personal_nodes).to(device=device, dtype=torch.long)
    feed_dict["final_node_embed"] = final_node_embed
    return feed_dict


def build_flatten_action_masks(state_tracker_field, mode, rollout=True):
    if rollout is True:
        rl_action_mask = np.concatenate(
            [[field[-1]["rl_action_mask"]] for field in state_tracker_field])
        if mode == "RuleWarmUp":
            warm_up_action_mask = np.concatenate(
                [[field[-1]["warm_up_action_mask"]] for field in state_tracker_field])
    else:
        rl_action_mask = np.concatenate(
            [[np.concatenate([[state["rl_action_mask"]] for state in field])] for field in state_tracker_field])
        if mode == "RuleWarmUp":
            warm_up_action_mask = np.concatenate(
                [[np.concatenate([[state["warm_up_action_mask"]] for state in field])] for field in
                 state_tracker_field])

    feed_dict = dict()
    feed_dict["rl_action_mask"] = torch.Tensor(rl_action_mask).to(device=device, dtype=torch.uint8)
    if mode == "RuleWarmUp":
        feed_dict["warm_up_action_mask"] = torch.Tensor(warm_up_action_mask).to(device=device,
                                                                                dtype=torch.uint8)

    return feed_dict


def build_hierarchy_action_masks(state_tracker_field, mode, rollout=True):
    """
    :param state_tracker_field:
    :param mode: RL or RuleWarmUp
    :param rollout: if rollout is False, * is the time_step.
    :return:
    rl_manager_action_mask: (batch_size, *, workers_num + 2)
    rl_workers_action_mask: (batch_size, *, workers_num, answer_node_num + 2)
    if mode == "RuleWarmUp"
        warm_up_manager_action_mask: (batch_size, *, workers_num + 2)
        warm_up_workers_action_mask: (batch_size, *, workers_num, answer_node_num + 2)
    """
    if rollout is True:
        rl_manager_action_mask = np.concatenate(
            [[field[-1]["rl_manager_action_mask"]] for field in state_tracker_field])
        rl_workers_action_mask = np.concatenate(
            [[field[-1]["rl_workers_action_mask"]] for field in state_tracker_field])
        if mode == "RuleWarmUp":
            warm_up_manager_action_mask = np.concatenate(
                [[field[-1]["warm_up_manager_action_mask"]] for field in state_tracker_field])
            warm_up_workers_action_mask = np.concatenate(
                [[field[-1]["warm_up_workers_action_mask"]] for field in state_tracker_field])
    else:
        rl_manager_action_mask = np.concatenate(
            [[np.concatenate([[state["rl_manager_action_mask"]] for state in field])] for field in state_tracker_field])
        rl_workers_action_mask = np.concatenate(
            [[np.concatenate([[state["rl_workers_action_mask"]] for state in field])] for field in state_tracker_field])
        if mode == "RuleWarmUp":
            warm_up_manager_action_mask = np.concatenate(
                [[np.concatenate([[state["warm_up_manager_action_mask"]] for state in field])] for field in
                 state_tracker_field])
            warm_up_workers_action_mask = np.concatenate(
                [[np.concatenate([[state["warm_up_workers_action_mask"]] for state in field])] for field in
                 state_tracker_field])

    feed_dict = dict()
    feed_dict["rl_manager_action_mask"] = torch.Tensor(rl_manager_action_mask).to(device=device, dtype=torch.uint8)
    feed_dict["rl_workers_action_mask"] = torch.Tensor(rl_workers_action_mask).to(device=device, dtype=torch.uint8)
    if mode == "RuleWarmUp":
        feed_dict["warm_up_manager_action_mask"] = torch.Tensor(warm_up_manager_action_mask).to(device=device,
                                                                                                dtype=torch.uint8)
        feed_dict["warm_up_workers_action_mask"] = torch.Tensor(warm_up_workers_action_mask).to(device=device,
                                                                                                dtype=torch.uint8)

    return feed_dict


def build_hierarchy_manager_inputs(manager_state, workers_state, policy_field, rollout=True):
    """
    :param manager_state: (batch_size, *, feature_size)
    :param workers_state: (batch_size, *, workers_num, feature_size)
    :return:
    manager_state: (batch_size, *, feature_size)
    workers_state: (batch_size, *, workers_num, feature_size)
    """
    personal_nodes = policy_field["manager_actions"]

    if rollout is False:
        time_step = manager_state.shape[1]
        personal_nodes = np.repeat(personal_nodes[:, np.newaxis], time_step, axis=1)

    feed_dict = dict()
    feed_dict["manager_state"] = manager_state
    feed_dict["workers_state"] = workers_state
    feed_dict["personal_nodes"] = torch.Tensor(personal_nodes).to(device=device, dtype=torch.long)
    return feed_dict


def build_flatten_manager_inputs(manager_state, policy_field, final_node_embed, rollout=True):
    answer_nodes = policy_field["actions"][:, :, 1]
    q_nodes = policy_field["actions"][:, :, 0]
    if rollout is False:
        time_step = manager_state.shape[1]
        answer_nodes = np.repeat(answer_nodes[:, np.newaxis], time_step, axis=1)
        q_nodes = np.repeat(q_nodes[:, np.newaxis], time_step, axis=1)
    feed_dict = dict()
    feed_dict["manager_state"] = manager_state
    feed_dict["answer_nodes"] = torch.Tensor(answer_nodes).to(device=device, dtype=torch.long)
    feed_dict["q_nodes"] = torch.Tensor(q_nodes).to(device=device, dtype=torch.long)
    feed_dict["graph_node_embedding"] = final_node_embed
    return feed_dict


def build_workers_inputs(workers_state, policy_field, final_node_embed, rollout=True):
    """
    :param workers_state: (batch_size, *, workers_num, feature_size)
    :param policy_field:
    :param final_node_embed: (batch_size, *, max_node_num, feature_size)
    :param rollout: if rollout is False, * is the time_step.
    :return:
    workers_state: (batch_size, *, workers_num, feature_size)
    answer_nodes: (batch_size, *, workers_num, answer_node_num)
    """
    answer_nodes = policy_field["workers_actions"]

    if rollout is False:
        time_step = workers_state.shape[1]
        answer_nodes = np.repeat(answer_nodes[:, np.newaxis], time_step, axis=1)

    feed_dict = dict()
    feed_dict["workers_state"] = workers_state
    feed_dict["answer_nodes"] = torch.Tensor(answer_nodes).to(device=device, dtype=torch.long)
    feed_dict["graph_node_embedding"] = final_node_embed
    return feed_dict
