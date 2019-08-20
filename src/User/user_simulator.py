# coding = utf-8
import random
import numpy as np
import copy
from src.User.knowlege_sampler import KnowledgeSampler
from src.config import User_Type_Weights, Personal_Information_Fraud_Weights, Known, UnKnown, ShowUnknown, FRAUD, \
    NON_FRAUD


class UserSimulator(object):
    def __init__(self, knowledge_sampler_filed):
        self.init_episode(knowledge_sampler_filed)

    def init_episode(self, knowledge_sampler_filed):
        self.knowledge_sampler_filed = knowledge_sampler_filed
        self.knowledge_sampler = KnowledgeSampler()
        self.user_type, feasible_personal_info_nodes = self.get_user_type()
        self.get_identity_state(feasible_personal_info_nodes)
        self.get_knowledge_sample_results()

    def get_user_type(self):
        feasible_personal_info_nodes = list()
        for q_node, a_edges in self.knowledge_sampler_filed["one_step_node_edges"].items():
            if len(a_edges) > 0:
                feasible_personal_info_nodes.append(int(q_node))

        type_weights = copy.deepcopy(User_Type_Weights)
        if len(feasible_personal_info_nodes) < 4:
            del type_weights["Type-4 Fraud"]
        if len(feasible_personal_info_nodes) < 3:
            del type_weights["Type-3 Fraud"]
        if len(feasible_personal_info_nodes) < 2:
            del type_weights["Type-2 Fraud"]
        if len(feasible_personal_info_nodes) < 1:
            del type_weights["Type-1 Fraud"]

        user_types = list(type_weights.keys())
        probs = np.asarray(list(type_weights.values()), dtype=np.float32)
        probs = probs / probs.sum()
        user_type = np.random.choice(user_types, p=probs.ravel())
        return user_type, feasible_personal_info_nodes

    def get_identity_state(self, feasible_personal_info_nodes):
        if self.user_type == "Non-Fraud":
            self.user_identity_state = NON_FRAUD
        else:
            self.user_identity_state = FRAUD

        identity_dict_keys = self.knowledge_sampler_filed["identity_dict_keys"]
        identity_dict_values = [NON_FRAUD for _ in identity_dict_keys]

        personal_information_fraud_weights = copy.deepcopy(Personal_Information_Fraud_Weights)
        include_keys = list()
        for node in feasible_personal_info_nodes:
            for k, v in self.knowledge_sampler_filed["personal_information"].items():
                if node == v:
                    include_keys.append(k)
                    break
        delete_keys = list(set(list(personal_information_fraud_weights.keys())) - set(include_keys))
        for k in delete_keys:
            del personal_information_fraud_weights[k]

        if self.user_type == "Type-4 Fraud":
            repeat_times = 4
        elif self.user_type == "Type-3 Fraud":
            repeat_times = 3
        elif self.user_type == "Type-2 Fraud":
            repeat_times = 2
        elif self.user_type == "Type-1 Fraud":
            repeat_times = 1
        else:
            repeat_times = 0

        fraud_items = list(personal_information_fraud_weights.keys())
        probs = np.asarray(list(personal_information_fraud_weights.values()), dtype=np.float32)
        probs = probs / probs.sum()
        sampled_fraud_items = list(np.random.choice(fraud_items, size=repeat_times, replace=False, p=probs.ravel()))

        for item in sampled_fraud_items:
            idx = identity_dict_keys.index(str(self.knowledge_sampler_filed["personal_information"][item]))
            identity_dict_values[idx] = FRAUD

        self.knowledge_sampler_filed["identity_dict_values"] = identity_dict_values
        self.user_sub_identity_state_dict = dict(zip(identity_dict_keys, identity_dict_values))

    def get_knowledge_sample_results(self):
        one_step_node_edges = self.knowledge_sampler_filed["one_step_node_edges"]
        adj_matrix = self.knowledge_sampler_filed["adj_matrix"]
        edge_se_freqs_matrix = self.knowledge_sampler_filed["edge_se_freqs_matrix"]
        edges = self.knowledge_sampler_filed["edges"]
        identity_dict_keys = self.knowledge_sampler_filed["identity_dict_keys"]
        identity_dict_values = self.knowledge_sampler_filed["identity_dict_values"]
        identity_dict = dict(zip(identity_dict_keys, identity_dict_values))

        results = self.knowledge_sampler.knowledge_sampler(edges, adj_matrix, edge_se_freqs_matrix,
                                                           one_step_node_edges, identity_dict)

        self.knowledge_sampler_filed["results"] = results

    def get_edge_idx(self, q_node, a_node):
        for i, (h, r, t) in enumerate(self.knowledge_sampler_filed["edges"]):
            if q_node == t and a_node == h:
                return i

    def answer(self, q_node, a_node):
        """  user will choose an answer from the candidates randomly if they don't know that. """
        if q_node is None and a_node is None:
            return None, None

        knowledge_state = self.knowledge_sampler_filed["results"][self.get_edge_idx(q_node, a_node)]
        if knowledge_state is True:
            user_answer = self.knowledge_sampler_filed["idx2node"][str(a_node)]
            answer_state = Known
        else:
            user_answer = ShowUnknown
            answer_state = UnKnown
        return user_answer, answer_state
