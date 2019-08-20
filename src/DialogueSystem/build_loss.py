# coding = utf-8
import torch
import torch.nn.functional as F
import numpy as np
from src.NNModule.agents import get_hierarchy_rl_returns, get_flatten_rl_returns
from src.config import device, EPS


def build_flatten_loss(state_tracker_field, agent_logits, agent_values, mode):
    """
    :param state_tracker_field:
    :param agent_logits: (batch, time_step, answer_node_num + 2)
    :param agent_values:  (batch, time_step)
    :param mode: RL or RuleWarmUp, if RuleWarmUp, the policy loss is the negative likelihood loss.
    :return: policy_loss, value_loss, entropy
    """
    rewards = np.concatenate(
        [[np.concatenate([[state["reward"]] for state in field])] for field in state_tracker_field])
    turn_masks = np.asarray([[state["episode_not_end"] for state in field] for field in state_tracker_field],
                            dtype=np.float32)
    rewards = torch.Tensor(rewards).to(device=device, dtype=torch.float32)
    turn_masks = torch.Tensor(turn_masks).to(device=device, dtype=torch.float32)
    returns = get_flatten_rl_returns(rewards, turn_masks)

    advantages = returns - agent_values

    # action index
    sample_idx = np.concatenate(
        [[np.concatenate([[state["sample_idx"]] for state in field])] for field in state_tracker_field])
    log_probs = F.log_softmax(agent_logits, dim=2)
    batch_size = log_probs.shape[0]
    time_step = log_probs.shape[1]
    selected_log_probs = log_probs.reshape(batch_size * time_step, -1)[
        torch.arange(end=batch_size * time_step, device=device), sample_idx.reshape(-1)
    ].reshape(batch_size, time_step)

    turns = turn_masks.sum(dim=1) + EPS

    if mode == "RL":
        policy_loss = - ((advantages.detach() * selected_log_probs * turn_masks).sum(1) / turns).sum() / batch_size
    elif mode == "RuleWarmUp":
        policy_loss = - ((selected_log_probs * turn_masks).sum(1) / turns).sum() / batch_size
    else:
        raise ValueError('Unknown mode.')
    value_loss = (((0.5 * (advantages ** 2)) * turn_masks).sum(dim=1) / turns).sum() / batch_size
    entropy = (((F.softmax(agent_logits, dim=2) * log_probs).sum(dim=2) * turn_masks).sum(
        dim=1) / turns).sum() / batch_size
    return policy_loss, value_loss, entropy


def build_hierarchy_loss(state_tracker_field, manager_logits, manager_values, workers_logits, workers_values, mode):
    """
    :param state_tracker_field:
    :param manager_logits: (batch_size, time_step, personal_node_num + 2)
    :param manager_values: (batch_size, time_step)
    :param workers_logits: (batch_size, time_step, personal_node_num, answer_node_num + 2)
    :param workers_values: (batch_size, time_step, personal_node_num)
    :param mode: RL or RuleWarmUp, if RuleWarmUp, the policy loss is the negative likelihood loss.
    :return:
    manager_policy_loss, workers_policy_loss,
    manager_value_loss, workers_value_loss,
    manager_avg_entropy, workers_avg_entropy
    """
    # Get the discounted rewards
    manager_rewards = np.concatenate(
        [[np.concatenate([[state["manager_reward"]] for state in field])] for field in state_tracker_field])
    workers_rewards = np.concatenate(
        [[np.concatenate([[state["workers_reward"]] for state in field])] for field in state_tracker_field])
    rewards = np.concatenate([manager_rewards, workers_rewards], axis=2)
    rewards = torch.Tensor(rewards).transpose(1, 2).to(device=device, dtype=torch.float32)

    reward_masks = np.concatenate(
        [[np.concatenate([[state["reward_mask"]] for state in field])] for field in state_tracker_field])
    reward_masks = torch.Tensor(reward_masks).transpose(1, 2).to(device=device, dtype=torch.uint8)

    # (batch_size, time_steps, 1 + personal_node_num)
    returns = get_hierarchy_rl_returns(rewards, reward_masks).transpose(1, 2)

    # get the values (batch_size, time_step, 1 + personal_node_num)
    values = torch.cat((manager_values.unsqueeze(2), workers_values), dim=2)

    # get the advantages (batch_size, time_step, 1 + personal_node_num)
    advantages = returns - values

    # Get the manager and workers sample index
    manager_sample_idx = np.concatenate(
        [[np.concatenate([[state["manager_sample_idx"]] for state in field])] for field in state_tracker_field])
    workers_sample_idx = np.concatenate(
        [[np.concatenate([[state["workers_sample_idx"]] for state in field])] for field in state_tracker_field])
    # (batch_size, time_steps)
    manager_sample_idx = torch.Tensor(manager_sample_idx).to(device=device, dtype=torch.long)
    # (batch_size, time_steps, personal_node_num)
    workers_sample_idx = torch.Tensor(workers_sample_idx).to(device=device, dtype=torch.long)

    # Get the policy masks (batch_size, time_steps, 1 + personal_node_num)
    policy_masks = np.concatenate(
        [[np.concatenate([[state["policy_mask"]] for state in field])] for field in state_tracker_field])
    policy_masks = torch.Tensor(policy_masks).to(device=device, dtype=torch.float32)

    # Get the episode over mask (batch_size, time_steps)
    turn_masks = np.asarray([[state["episode_not_end"] for state in field] for field in state_tracker_field],
                            dtype=np.float32)
    turn_masks = torch.Tensor(turn_masks).to(device=device, dtype=torch.float32)

    # get the manager selected log probs (batch_size, time_step, 1)
    manager_log_probs = F.log_softmax(manager_logits, dim=2)
    batch_size = manager_log_probs.shape[0]
    time_step = manager_log_probs.shape[1]
    manager_selected_log_probs = manager_log_probs.reshape(batch_size * time_step, -1)[
        torch.arange(end=batch_size * time_step, device=device), manager_sample_idx.reshape(-1)
    ].reshape(batch_size, time_step, 1)

    # get the workers selected log probs (batch, time_step, personal_node_num)
    workers_log_probs = F.log_softmax(workers_logits, dim=3)
    batch_size = workers_log_probs.shape[0]
    time_step = workers_log_probs.shape[1]
    personal_node_num = workers_log_probs.shape[2]
    workers_selected_log_probs = workers_log_probs.reshape(batch_size * time_step * personal_node_num, -1)[
        torch.arange(end=batch_size * time_step * personal_node_num, device=device), workers_sample_idx.reshape(-1)
    ].reshape(batch_size, time_step, personal_node_num)

    # selected log probs (batch, time_step, 1 + personal_node_num)
    log_selected_probs = torch.cat((manager_selected_log_probs, workers_selected_log_probs), dim=2)

    # turns average (batch, 1 + personal_node_num)
    turns = (policy_masks * turn_masks.unsqueeze(2)).sum(dim=1) + EPS

    # valid workers num (batch,)
    valid_workers_num = np.concatenate([[field[-1]["valid_workers_num"]] for field in state_tracker_field])
    valid_workers_num = torch.Tensor(valid_workers_num).to(device=device, dtype=torch.float32)

    # the policy loss
    if mode == "RL":
        policy_loss = - ((advantages.detach() * log_selected_probs * policy_masks) * turn_masks.unsqueeze(2)).sum(
            dim=1) / turns
    elif mode == "RuleWarmUp":
        policy_loss = - ((log_selected_probs * policy_masks) * turn_masks.unsqueeze(2)).sum(dim=1) / turns
    else:
        raise ValueError('Unknown mode.')
    manager_policy_loss = policy_loss[:, 0].sum(0) / batch_size
    workers_policy_loss = (policy_loss[:, 1:].sum(1) / valid_workers_num).sum(0) / batch_size

    # the value loss
    value_loss = ((policy_masks * (0.5 * (advantages ** 2))) * turn_masks.unsqueeze(2)).sum(dim=1) / turns
    manager_value_loss = value_loss[:, 0].sum(0) / batch_size
    workers_value_loss = (value_loss[:, 1:].sum(1) / valid_workers_num).sum(0) / batch_size

    # entropy regularization
    manager_entropy = (((F.softmax(manager_logits, dim=2) * manager_log_probs).sum(dim=2)) * (
            turn_masks * policy_masks[:, :, 0])).sum(dim=1, keepdim=True)  # (batch, 1)
    workers_entropy = (((F.softmax(workers_logits, dim=3) * workers_log_probs).sum(dim=3)) * (
            turn_masks.unsqueeze(2) * policy_masks[:, :, 1:])).sum(dim=1)  # (batch, personal_node_num)
    entropy = (torch.cat((manager_entropy, workers_entropy), dim=1)) / turns
    manager_avg_entropy = entropy[:, 0].sum(0) / batch_size
    workers_avg_entropy = (entropy[:, 1:].sum(1) / valid_workers_num).sum(0) / batch_size

    return manager_policy_loss, workers_policy_loss, manager_value_loss, workers_value_loss, manager_avg_entropy, workers_avg_entropy
