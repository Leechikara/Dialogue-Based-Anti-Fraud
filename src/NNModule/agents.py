# coding = utf-8
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from src.NNModule.utils import Attn, batch_embedding_lookup
from src.config import FRAUD, NON_FRAUD, ManagerRewardDiscount, WorkerRewardDiscount, EPS, Pad_Query_Node


class Agent(nn.Module):
    def __init__(self, score_method, agent_state_size, answer_node_emb_size):
        super(Agent, self).__init__()
        self.policy_network = Attn(score_method, answer_node_emb_size, agent_state_size)
        self.value_network = nn.Sequential(
            nn.Linear(agent_state_size, agent_state_size // 2),
            nn.Tanh(),
            nn.Linear(agent_state_size // 2, 1)
        )
        # two vector to represent the fraud and non-fraud actions
        self.fraud_embed = Parameter(torch.Tensor(answer_node_emb_size))
        self.fraud_embed.data.uniform_(-1, 1)
        self.non_fraud_embed = Parameter(torch.Tensor(answer_node_emb_size))
        self.non_fraud_embed.data.uniform_(-1, 1)

    def forward(self, agent_state, answer_nodes, graph_node_embedding):
        """
        :param agent_state: (batch_size, agent_state_size)
        :param answer_nodes: (batch_size, answer_node_num)
        :param graph_node_embedding: (batch_size, node_num, node_feature_size)
        :return:
        values: (batch_size,)
        logits: (batch, answer_node_num + 2)
        """
        values = self.value_network(agent_state).squeeze(-1)

        batch_size = agent_state.shape[0]
        answer_node_embedding = batch_embedding_lookup(graph_node_embedding, answer_nodes)
        actions_embedding = torch.cat((answer_node_embedding,
                                       self.fraud_embed.repeat(batch_size, 1, 1),
                                       self.non_fraud_embed.repeat(batch_size, 1, 1)),
                                      dim=1)
        logits = self.policy_network(actions_embedding, agent_state)

        return values, logits


class Manager(nn.Module):
    def __init__(self, score_method, manager_state_size, worker_state_size):
        super(Manager, self).__init__()
        self.policy_network = Attn(score_method, worker_state_size, manager_state_size)
        self.value_network = nn.Sequential(
            nn.Linear(manager_state_size, manager_state_size // 2),
            nn.Tanh(),
            nn.Linear(manager_state_size // 2, 1)
        )

        # two vector to represent the fraud and non-fraud actions
        self.fraud_embed = Parameter(torch.Tensor(worker_state_size))
        self.fraud_embed.data.uniform_(-1, 1)
        self.non_fraud_embed = Parameter(torch.Tensor(worker_state_size))
        self.non_fraud_embed.data.uniform_(-1, 1)

    def forward(self, manager_state, workers_state):
        """
        :param manager_state: (batch_size, manager_state_size)
        :param workers_state: (batch_size, personal_node_num, worker_sate_size)
        :return:
        values: (batch_size,)
        logits: (batch_size, personal_node_num + 2)
        """
        values = self.value_network(manager_state).squeeze(-1)

        batch_size = manager_state.shape[0]
        actions_embedding = torch.cat((workers_state,
                                       self.fraud_embed.repeat(batch_size, 1, 1),
                                       self.non_fraud_embed.repeat(batch_size, 1, 1)),
                                      dim=1)
        logits = self.policy_network(actions_embedding, manager_state)

        return values, logits


class Workers(nn.Module):
    def __init__(self, score_method, worker_state_size, answer_node_emb_size):
        super(Workers, self).__init__()
        self.policy_networks = Attn(score_method, answer_node_emb_size, worker_state_size)
        self.value_network = nn.Sequential(
            nn.Linear(worker_state_size, worker_state_size // 2),
            nn.Tanh(),
            nn.Linear(worker_state_size // 2, 1)
        )

        # two vector to represent the fraud and non-fraud actions
        self.fraud_embed = Parameter(torch.Tensor(answer_node_emb_size))
        self.fraud_embed.data.uniform_(-1, 1)
        self.non_fraud_embed = Parameter(torch.Tensor(answer_node_emb_size))
        self.non_fraud_embed.data.uniform_(-1, 1)

    def forward(self, workers_state, answer_nodes, graph_node_embedding):
        """
        :param workers_state: (batch_size, personal_node_num, worker_state_size)
        :param answer_nodes: (batch_size, personal_node_num, answer_node_num)
        :param graph_node_embedding: (batch_size, node_num, node_feature_size)
        :return:
        values: (batch_size, personal_node_num)
        logits: (batch_size, personal_node_num, answer_node_num + 2)
        """
        values = self.value_network(workers_state).squeeze(-1)

        batch_size = answer_nodes.shape[0]
        personal_node_num = answer_nodes.shape[1]
        answer_node_num = answer_nodes.shape[2]

        answer_nodes = answer_nodes.reshape(batch_size, -1)
        answer_node_embedding = batch_embedding_lookup(graph_node_embedding, answer_nodes)
        answer_node_embedding = answer_node_embedding.reshape(batch_size, personal_node_num, answer_node_num, -1)
        actions_embedding = torch.cat((answer_node_embedding,
                                       self.fraud_embed.repeat(batch_size, personal_node_num, 1, 1),
                                       self.non_fraud_embed.repeat(batch_size, personal_node_num, 1, 1)),
                                      dim=2)

        workers_state = workers_state.reshape(batch_size * personal_node_num, -1)
        actions_embedding = actions_embedding.reshape(batch_size * personal_node_num, answer_node_num + 2, -1)

        logits = self.policy_networks(actions_embedding, workers_state)
        logits = logits.reshape(batch_size, personal_node_num, -1)
        return values, logits


def sample_from_prob_matrix(prob_matrix, sample_flag):
    """
    Sample n times based on prob matrix. The prob in i-th experiments is the i-th row of prob matrix.
    :param prob_matrix: (n_times, m_items)
    :param sample_flag: str, max, random, sample
    :return: choices: (n_times,)
    """
    if sample_flag == "sample":
        cumulative_prob = prob_matrix.cumsum(axis=1)
        uniform = np.random.rand(len(cumulative_prob), 1)
        choices = (uniform < cumulative_prob).argmax(axis=1)
    elif sample_flag == "max":
        choices = prob_matrix.argmax(axis=1)
    elif sample_flag == "random":
        new_prob_matrix = np.asarray(prob_matrix > 1000 * EPS, dtype=np.float32)
        items_matrix = new_prob_matrix.sum(axis=1, keepdims=True) + EPS
        choices = sample_from_prob_matrix(new_prob_matrix / items_matrix, sample_flag="sample")
    else:
        raise ValueError('Unknown Sample Flag.')
    return choices


def sample_hierarchy_rl(manager_action_probs, workers_action_probs, manager_actions, workers_actions, sample_flag):
    """
    Assume all inputs are np.array
    :param manager_action_probs: (batch_size, personal_node_num + 2)
    :param workers_action_probs: (batch_size, personal_node_num, answer_node_num + 2)
    :param manager_actions: (batch_size, personal_node_num)
    :param workers_actions: (batch_size, personal_node_num, answer_node_num)
    :param sample_flag:
    :return:
    manager_sample_idx: (batch_size,)
    manager_sample_result: (batch_size,)
    workers_sample_idx: (batch_size, personal_node_num)
    workers_sample_result: (batch_size, personal_node_num)
    """
    manager_sample_idx = sample_from_prob_matrix(manager_action_probs, sample_flag)

    batch_size = workers_action_probs.shape[0]
    personal_node_num = workers_action_probs.shape[1]
    workers_action_probs = workers_action_probs.reshape(batch_size * personal_node_num, -1)
    workers_sample_idx = sample_from_prob_matrix(workers_action_probs, sample_flag)

    manager_terminal_actions = np.tile(np.asarray([FRAUD, NON_FRAUD]), (batch_size, 1))
    manager_actions = np.concatenate((manager_actions, manager_terminal_actions), axis=1)
    manager_sample_result = manager_actions[np.arange(manager_actions.shape[0]), manager_sample_idx]

    worker_terminal_actions = np.tile(np.asarray([FRAUD, NON_FRAUD]), (batch_size, personal_node_num, 1))
    workers_actions = np.concatenate((workers_actions, worker_terminal_actions), axis=2)
    workers_actions = workers_actions.reshape(batch_size * personal_node_num, -1)
    workers_sample_result = workers_actions[np.arange(workers_actions.shape[0]), workers_sample_idx]

    workers_sample_result = workers_sample_result.reshape(batch_size, personal_node_num)
    workers_sample_idx = workers_sample_idx.reshape(batch_size, personal_node_num)

    return manager_sample_idx, manager_sample_result, workers_sample_idx, workers_sample_result


def sample_flatten_rl(action_probs, answer_nodes, query_nodes, sample_flag):
    """
    Assume all inputs are np.array
    :param action_probs: (batch_size, answer_node_num + 2)
    :param answer_nodes: (batch_size, answer_node_num)
    :param query_nodes: (batch_size, answer_node_num)
    :param sample_flag:
    :return:
    sample_idx: (batch_size,)
    sample_content: (batch_size, 2)
    """
    sample_idx = sample_from_prob_matrix(action_probs, sample_flag)
    batch_size = action_probs.shape[0]

    terminal_actions = np.tile(np.asarray([FRAUD, NON_FRAUD]), (batch_size, 1))
    pad_query_nodes = np.tile(np.asarray([Pad_Query_Node, Pad_Query_Node]), (batch_size, 1))
    answer_nodes = np.concatenate((answer_nodes, terminal_actions), axis=1)
    query_nodes = np.concatenate((query_nodes, pad_query_nodes), axis=1)
    sample_query = query_nodes[np.arange(query_nodes.shape[0]), sample_idx]
    sample_result = answer_nodes[np.arange(answer_nodes.shape[0]), sample_idx]

    sample_content = np.concatenate((sample_query[:, np.newaxis], sample_result[:, np.newaxis]), axis=1)

    return sample_idx, sample_content


def get_hierarchy_rl_returns(rewards, masks):
    """
    :param rewards: (batch_size, 1 + workers_num, time_steps)
    :param masks: (batch_size, 1 + workers_num, time_steps)
    :return: returns: (batch_size, 1 + workers_num, time_steps)
    """
    masks = masks.to(torch.uint8)

    gamma = torch.zeros_like(rewards)
    gamma[:, 0, :] = ManagerRewardDiscount
    gamma[:, 1:, :] = WorkerRewardDiscount
    gamma[1 - masks] = torch.ones_like(rewards)[1 - masks]

    batch_size = rewards.shape[0]
    agent_num = rewards.shape[1]
    time_steps = rewards.shape[2]
    rewards = rewards.reshape(batch_size * agent_num, -1)
    masks = masks.reshape(batch_size * agent_num, -1)
    gamma = gamma.reshape(batch_size * agent_num, -1)

    returns = torch.zeros_like(rewards)
    running_returns = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(0, time_steps)):
        running_returns = rewards[:, t] * masks[:, t].to(running_returns.dtype) + gamma[:, t] * running_returns
        returns[:, t] = running_returns

    returns = returns.reshape(batch_size, agent_num, time_steps)
    masks = masks.reshape(batch_size, agent_num, time_steps)

    # Mask the invalid items
    returns = returns * masks.to(returns.dtype)

    return returns


def get_flatten_rl_returns(rewards, masks):
    """
    :param rewards: (batch_size, time_steps)
    :param masks: (batch_size, time_steps)
    :return: returns: (batch_size, time_steps)
    """
    gamma = ManagerRewardDiscount
    time_steps = rewards.shape[1]

    returns = torch.zeros_like(rewards)
    running_returns = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(0, time_steps)):
        running_returns = rewards[:, t] * masks[:, t].to(running_returns.dtype) + gamma * running_returns
        returns[:, t] = running_returns

    # Mask the invalid items
    returns = returns * masks.to(returns.dtype)
    return returns
