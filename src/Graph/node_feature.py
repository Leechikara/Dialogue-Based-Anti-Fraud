# coding = utf-8
from itertools import chain
import os
import json
import math
import numpy as np
from src.config import DATA_ROOT, DEGREE_FILED, SE_FREQS_FILED, DYNAMIC_FEATURE_SIZE

with open(os.path.join(DATA_ROOT, "se_freqs_bins.json"), "r") as f:
    se_freqs_bins = json.load(f)


class NodeFeature(object):
    def __init__(self):
        self.se_freqs_bins = se_freqs_bins

    def static_feature(self,
                       nodes,
                       personal_information,
                       one_step_nodes,
                       node_se_freqs,
                       node_degree,
                       node_in_degree,
                       node_out_degree):
        """  Get feature before the dialogue.  """
        answer_nodes = list(chain(*one_step_nodes.values()))
        static_feature_matrix = list()
        for node, se_freqs, degree, in_degree, out_degree in zip(nodes,
                                                                 node_se_freqs,
                                                                 node_degree,
                                                                 node_in_degree,
                                                                 node_out_degree):
            vector = list()

            # the personal information type of this node
            for value in personal_information.values():
                if node == value:
                    vector.append(1)
                else:
                    vector.append(0)

            # if the node is the answer node
            if node in answer_nodes:
                vector.append(1)
            else:
                vector.append(0)

            for i, max_se_freqs in enumerate(self.se_freqs_bins):
                if math.log(se_freqs + 1) <= max_se_freqs:
                    vector.extend(self.one_hot(i, SE_FREQS_FILED))
                    break

            vector.extend(self.one_hot(degree, DEGREE_FILED))
            # vector.extend(self.one_hot(in_degree, IN_DEGREE_FILED))
            # vector.extend(self.one_hot(out_degree, OUT_DEGREE_FILED))
            static_feature_matrix.append(vector)
        return static_feature_matrix

    @staticmethod
    def one_hot(idx, max_length):
        v = [0 for _ in range(max_length)]
        if idx > max_length - 1:
            v[-1] = 1
        else:
            v[idx] = 1
        return v

    @staticmethod
    def dialogue_feature(max_node_num,
                         nodes,
                         explored_nodes,
                         last_turn_q_node,
                         last_turn_a_node,
                         not_explored_nodes,
                         known_nodes,
                         unknown_nodes,
                         not_answered_nodes):
        """  Get feature during the dialogue. Support batch.  """
        dialogue_feature_matrix = np.zeros((max_node_num, DYNAMIC_FEATURE_SIZE))

        for node in nodes:
            vector = list()
            if node in explored_nodes:
                vector.append(1)
            else:
                vector.append(0)

            if node == last_turn_q_node:
                vector.append(1)
            else:
                vector.append(0)

            if node == last_turn_a_node:
                vector.append(1)
            else:
                vector.append(0)

            if node in not_explored_nodes:
                vector.append(1)
            else:
                vector.append(0)

            if node in known_nodes:
                vector.append(1)
            else:
                vector.append(0)

            if node in unknown_nodes:
                vector.append(1)
            else:
                vector.append(0)

            if node in not_answered_nodes:
                vector.append(1)
            else:
                vector.append(0)

            dialogue_feature_matrix[node] = np.asarray(vector, dtype=np.float32)

        return dialogue_feature_matrix
