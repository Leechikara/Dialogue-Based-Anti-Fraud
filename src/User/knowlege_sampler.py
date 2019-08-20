# coding  = utf-8
import random
import json
import os
import numpy as np
import math
from src.config import DATA_ROOT, Known, UnKnown, NotClear, FRAUD, NON_FRAUD

with open(os.path.join(DATA_ROOT, "prob_bin.json"), "r") as f:
    prob_bin = json.load(f)


class KnowledgeSampler(object):
    def __init__(self):
        self.prob_bin = prob_bin

    def knowledge_sampler(self, edges, adj_matrix, edge_se_freqs_matrix, one_step_node_edges, identity_dict):
        """
        identity_dict: type: dict. key: idx of one step node to User, value is Fraud or NON_FRAUD.
        if the users are non-fraud applicants, all values in identity_dict are NON_FRAUD.
        If the users are type-0 fraud applicants, all values in identity_dict are Fraud.
        If the users are type-1 fraud applicants, only one value in identity_dict is NON_FRAUD.

        :return: bool list. the length of results is the same as the edge num.
        """
        known_matrix = self.user_knowledge_sample(adj_matrix, edge_se_freqs_matrix, one_step_node_edges, identity_dict)
        edge_sample_results = list()
        for h, _, t in edges:
            edge_sample_results.append(bool(known_matrix[h][t]))
        return edge_sample_results

    def user_knowledge_sample(self, adj_matrix, edge_se_freqs_matrix, one_step_node_edges, identity_dict):
        """  the sampling and calibration algorithm  """
        init_known_matrix = self.edge_sample(edge_se_freqs_matrix, one_step_node_edges, identity_dict)
        known_matrix = self.calibration(adj_matrix, init_known_matrix)
        return known_matrix

    def sample(self, identity_state, se_freqs):
        """identity_state: FRAUD or NON_FRAUD"""
        if identity_state == NON_FRAUD:
            prob_bin = self.prob_bin["non-fraud"]
        elif identity_state == FRAUD:
            prob_bin = self.prob_bin["fraud"]

        for max_bin, prob in prob_bin.items():
            if math.log(se_freqs + 1) < float(max_bin):
                p = random.random()
                if p < float(prob):
                    return Known
                else:
                    return UnKnown
        return Known

    def edge_se_freqs_sample(self, node_idx1, node_idx2, se_freqs, one_step_node_edges, identity_dict):
        for node, edges in one_step_node_edges.items():
            for h, r, t in edges:
                if h == node_idx1 and t == node_idx2:
                    identity_state = identity_dict[node]
                    return self.sample(identity_state, se_freqs)

    def edge_sample(self, edge_se_freqs_matrix, one_step_node_edges, identity_dict):
        node_num = edge_se_freqs_matrix.shape[0]
        known_matrix = np.full(edge_se_freqs_matrix.shape, NotClear)
        for i in range(node_num):
            for j in range(node_num):
                if known_matrix[i][j] != NotClear:
                    assert known_matrix[i][j] == known_matrix[j][i]
                    continue

                if edge_se_freqs_matrix[i][j] == float("inf"):
                    assert i == j
                    known_matrix[i][j] = Known
                elif edge_se_freqs_matrix[i][j] == float("-inf"):
                    if edge_se_freqs_matrix[j][i] != float("-inf"):
                        se_freqs = edge_se_freqs_matrix[j][i]
                        known_matrix[j][i] = self.edge_se_freqs_sample(j, i, se_freqs,
                                                                       one_step_node_edges,
                                                                       identity_dict)
                        known_matrix[i][j] = known_matrix[j][i]
                    else:
                        known_matrix[i][j] = UnKnown
                        known_matrix[j][i] = UnKnown
                elif edge_se_freqs_matrix[i][j] != float("-inf"):
                    if edge_se_freqs_matrix[j][i] != float("-inf"):
                        assert edge_se_freqs_matrix[j][i] == edge_se_freqs_matrix[i][j]
                    se_freqs = edge_se_freqs_matrix[i][j]
                    known_matrix[i][j] = self.edge_se_freqs_sample(i, j, se_freqs,
                                                                   one_step_node_edges,
                                                                   identity_dict)
                    known_matrix[j][i] = known_matrix[i][j]

        return known_matrix == np.full(known_matrix.shape, Known)

    @staticmethod
    def calibration(adj_matrix, init_known_matrix):
        """  calibration algorithm is similar to arrive matrix  """
        known_matrix = init_known_matrix.copy()
        node_num = known_matrix.shape[0]
        for _ in range(node_num):
            new_known_matrix = known_matrix.dot(init_known_matrix) * adj_matrix
            if np.equal(new_known_matrix, known_matrix).all():
                break
            known_matrix = new_known_matrix.copy()
        return known_matrix
