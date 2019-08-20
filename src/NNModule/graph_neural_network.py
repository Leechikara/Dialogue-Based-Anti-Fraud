# coding = utf-8
import torch
import torch.nn as nn
from src.NNModule.utils import batch_embedding_lookup
from src.config import EPS


class GNN(nn.Module):
    """ A pytorch implementation of Message Passing Network """

    def __init__(self,
                 node_emb_size_list,
                 msg_agg):
        super(GNN, self).__init__()

        self.mp_iters = len(node_emb_size_list) - 1
        self.linear_cells = nn.ModuleList([nn.Linear(in_features, out_features) for in_features, out_features in
                                           zip(node_emb_size_list[:-1], node_emb_size_list[1:])])
        self.msg_agg = msg_agg

    def embed_edge(self, node_embedding, edges, iter_idx):
        """
        Compute embedding of a edge (sender_node, relation_label).
        :param node_embedding: (batch_size, max_node_num, feature_size)
        :param edges: each edge is a tuple of (sender_node, relation_label, receiver_node).
        (batch_size, max_edge_num, 3)
        :param iter_idx: the iter_idx-th iteration of graph
        :return: edge_embeds: (batch_size, max_edge_num, feature_size)
        """
        sender_embeds = batch_embedding_lookup(node_embedding, edges[:, :, 0])
        edge_embeds = torch.tanh(self.linear_cells[iter_idx](sender_embeds))
        return edge_embeds

    def pass_message(self, node_edges, node_edge_mask, edge_embeds):
        """
        Compute new node embedding by summing edge embeds (message) of neighboring nodes.
        :param node_edges: ids of neighboring edges of each node where id is row index in edge_embeds
        (batch_size, max_node_num, max_node_edge_num)
        :param node_edge_mask: mask for node_edges. (batch_size, max_node_num, max_node_edge_num)
        :param edge_embeds: (batch_size, max_edge_num, feature_size)
        :return:
        """
        num_neighbors = torch.sum(node_edge_mask.to(torch.float32), 2, keepdim=True) + EPS

        shape = node_edges.shape
        batch_size = shape[0]
        node_num = shape[1]
        edge_embed_size = edge_embeds.shape[-1]

        # Gather neighboring edge embeddings
        neighbors = torch.reshape(node_edges, (batch_size, -1))  # (batch_size, max_node_num * max_node_edge_num)
        embeds = batch_embedding_lookup(edge_embeds,
                                        neighbors)  # (batch_size, max_node_num * max_node_edge_num, feature_size)
        embeds = torch.reshape(embeds, (batch_size, node_num, -1, edge_embed_size))
        mask = torch.unsqueeze(node_edge_mask, 3)  # (batch_size, max_node_num, max_node_edge_num, 1)
        embeds = embeds * mask.to(embeds.dtype)

        # (batch_size, max_node_num, feature_size)
        if self.msg_agg == 'sum':
            new_node_embed = torch.sum(embeds, 2)
        elif self.msg_agg == 'avg':
            new_node_embed = torch.sum(embeds, 2) / num_neighbors
        elif self.msg_agg == 'max':
            new_node_embed, _ = torch.max(embeds, 2)
        else:
            raise ValueError('Unknown message aggregation method')

        return new_node_embed

    def mp(self, curr_node_embedding, edges, iter_idx, node_edges, node_edge_mask):
        edge_embeds = self.embed_edge(curr_node_embedding, edges, iter_idx)
        new_node_embed = self.pass_message(node_edges, node_edge_mask, edge_embeds)
        return new_node_embed

    def forward(self, initial_node_embed, edges, node_edges, node_edge_mask):
        """
        :param initial_node_embed: (batch_size, max_node_num, feature_size)
        :param edges: (batch_size, max_edge_num, 3)
        :param node_edges: (batch_size, max_node_num, max_node_edge_num)
        :param node_edge_mask: (batch_size, max_node_num, max_node_edge_num)
        :return: final_node_embed: (batch_size, max_node_num, feature_size)
        """
        node_embed_list = [initial_node_embed]
        for iter_idx in range(self.mp_iters):
            node_embed_list.append(self.mp(node_embed_list[-1], edges, iter_idx, node_edges, node_edge_mask))
        final_node_embed = torch.cat(node_embed_list, 2)
        return final_node_embed


class WorkersStateTracker(nn.Module):
    """ Concat personal nodes embedding and hand-crafted features to get the workers dialogue state """

    def __init__(self):
        super(WorkersStateTracker, self).__init__()

    def forward(self, known_one_hot, unknown_one_hot, known_differ_one_hot, workers_qa_turn_one_hot,
                workers_max_qa_turn_one_hot, personal_nodes, final_node_embed):
        """
        :param known_one_hot: (batch_size, personal_node_num, feature_size)
        :param unknown_one_hot: (batch_size, personal_node_num, feature_size)
        :param known_differ_one_hot: (batch_size, personal_node_num, feature_size)
        :param workers_qa_turn_one_hot: (batch_size, personal_node_num, feature_size)
        :param workers_max_qa_turn_one_hot: (batch_size, personal_node_num, feature_size)
        :param personal_nodes: (batch_size, personal_node_num)
        :param final_node_embed: (batch_size, feature_size)
        :return: workers_state: (batch_size, personal_node_num, feature_size)
        """
        personal_node_embed = batch_embedding_lookup(final_node_embed, personal_nodes)

        workers_state = torch.cat((known_one_hot,
                                   unknown_one_hot,
                                   known_differ_one_hot,
                                   workers_qa_turn_one_hot,
                                   workers_max_qa_turn_one_hot,
                                   personal_node_embed),
                                  2)
        return workers_state


class ManagerStateTracker(nn.Module):
    """ Aggregate personal nodes embedding and hand-crafted features to get the manager dialogue state """

    def __init__(self, personal_node_emb_size, manager_agg_size, msg_agg):
        super(ManagerStateTracker, self).__init__()

        self.msg_agg = msg_agg
        self.transfer = nn.Sequential(nn.Linear(personal_node_emb_size, manager_agg_size),
                                      nn.Tanh())

    def forward(self, feasible_personal_info_nodes, workers_decision,
                known_one_hot, unknown_one_hot, known_differ_one_hot,
                total_qa_turn_one_hot, personal_nodes, final_node_embed):
        """
        :param feasible_personal_info_nodes: (batch_size, personal_node_num)
        :param workers_decision: (batch_size, personal_node_num, 2)
        :param known_one_hot: (batch_size, personal_node_num, feature_size)
        :param unknown_one_hot: (batch_size, personal_node_num, feature_size)
        :param known_differ_one_hot: (batch_size, personal_node_num, feature_size)
        :param total_qa_turn_one_hot: (batch_size, feature_size)
        :param personal_nodes: (batch_size, personal_node_num)
        :param final_node_embed: (batch_size, max_node_num, feature_size)
        :return: manager_state: (batch_size, feature_size)
        """
        batch_size = personal_nodes.shape[0]
        personal_node_num = personal_nodes.shape[1]

        personal_node_embed = batch_embedding_lookup(final_node_embed, personal_nodes)
        transferred_personal_node_embed = self.transfer(personal_node_embed)

        if self.msg_agg == 'sum':
            agg_state = torch.sum(transferred_personal_node_embed, 1)
        elif self.msg_agg == 'avg':
            agg_state = torch.sum(transferred_personal_node_embed, 1) / personal_node_num
        elif self.msg_agg == 'max':
            agg_state, _ = torch.max(transferred_personal_node_embed, 1)
        else:
            raise ValueError('Unknown message aggregation method')

        manager_state = torch.cat((feasible_personal_info_nodes,
                                   workers_decision.reshape(batch_size, -1),
                                   known_one_hot.reshape(batch_size, -1),
                                   unknown_one_hot.reshape(batch_size, -1),
                                   known_differ_one_hot.reshape(batch_size, -1),
                                   total_qa_turn_one_hot,
                                   agg_state), 1)
        return manager_state
