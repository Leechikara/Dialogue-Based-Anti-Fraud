# coding = utf-8
import copy
import numpy as np
import os
import json
import random
from src.config import EdgePad, DATA_ROOT, PLACE_HOLDER, STATIC_FEATURE_SIZE, MaxWorkerExploringTimeStep
from itertools import chain


class GraphPreprocess(object):
    def __init__(self, data_set_dir="preprocessed_graphs", batch_size=32):
        # preprocess data set
        self.batch_size = batch_size
        self.train_set = self.split_batch(data_set_dir, "train.json", batch_size)
        self.test_set = self.split_batch(data_set_dir, "test.json", batch_size)
        self.dev_set = self.split_batch(data_set_dir, "dev.json", batch_size)

    def split_batch(self, data_dir, data_file, batch_size):
        with open(os.path.join(DATA_ROOT, data_dir, data_file), "r") as f:
            data_set = json.load(f)

        for _ in range(batch_size - len(data_set) % batch_size):
            data_set.append(copy.deepcopy(random.choice(data_set)))
        data_set.sort(key=lambda graph: len(graph["edges"]))

        split_data_set = list()
        for batch_idx in range(len(data_set) // batch_size):
            batch = list()
            for i in range(batch_size):
                batch.append(data_set[batch_idx * batch_size + i])
            split_data_set.append(self.preprocess_batch(batch))
        return split_data_set

    @staticmethod
    def get_graph_embed_filed(batch):
        """  Used in Graph Embed  """
        max_node_num = 0
        max_edge_num = 0
        max_node_edge_num = 0
        max_answer_node_num = 0
        max_qa_pair_num = 0
        for graph in batch:
            current_node_num = len(graph["nodes"])
            current_edge_num = len(graph["edges"])
            current_max_node_edge_num = max(map(len, graph["node_edges"]))
            current_max_answer_node_num = max(map(len, list(graph["one_step_nodes"].values())))
            current_qa_pair_num = sum(map(len, list(graph["one_step_nodes"].values())))

            if max_node_num < current_node_num:
                max_node_num = current_node_num
            if max_edge_num < current_edge_num:
                max_edge_num = current_edge_num
            if max_node_edge_num < current_max_node_edge_num:
                max_node_edge_num = current_max_node_edge_num
            if max_answer_node_num < current_max_answer_node_num:
                max_answer_node_num = current_max_answer_node_num
            if max_qa_pair_num < current_qa_pair_num:
                max_qa_pair_num = current_qa_pair_num

        # add 1 for node pad
        # the padded node idx is the current_node_num
        # but pad with range(current_node_num, max_node_num)
        max_node_num += 1

        # add 1 for edge pad
        # the padded edge idx is the current_edge_num
        # but pad with EdgePad
        max_edge_num += 1

        batch_node_num = list()
        batch_qa_pair_num = list()
        batch_personal_nodes = list()
        batch_feasible_personal_info_nodes = list()
        batch_static_feature = list()
        batch_edges = list()
        batch_node_edge_mask = list()
        batch_node_edges = list()

        for graph in batch:
            # **************** personal_nodes and node static feature ****************
            current_node_num = len(graph["nodes"])
            batch_node_num.append(current_node_num)
            current_qa_pair_num = sum(map(len, list(graph["one_step_nodes"].values())))
            batch_qa_pair_num.append(current_qa_pair_num)

            personal_nodes = np.asarray(graph["personal_nodes"], dtype=np.int32)
            batch_personal_nodes.append(personal_nodes)

            feasible_personal_info_nodes = np.zeros((len(graph["personal_nodes"]),), dtype=np.float32)
            for worker_idx, answer_nodes in graph["one_step_nodes"].items():
                if len(answer_nodes) > 0:
                    feasible_personal_info_nodes[graph["personal_nodes"].index(int(worker_idx))] = 1
            batch_feasible_personal_info_nodes.append(feasible_personal_info_nodes)

            original_static_feature = np.asarray(graph["static_feature"], dtype=np.float32)
            static_feature = np.zeros((max_node_num, STATIC_FEATURE_SIZE), dtype=np.float32)
            static_feature[:original_static_feature.shape[0]] = original_static_feature
            batch_static_feature.append(static_feature)

            # **************** edges ****************
            edges = np.zeros((max_edge_num, 3), dtype=np.int32)
            current_edge_num = len(graph["edges"])
            edges[:current_edge_num] = np.asarray(graph["edges"], dtype=np.int32)
            edges[current_edge_num:] = np.tile(np.asarray(EdgePad, dtype=np.int32),
                                               (max_edge_num - current_edge_num, 1))
            batch_edges.append(edges)

            # **************** pad node_edges and get mask ****************
            current_node_edge_num = map(len, graph["node_edges"])
            node_edge_mask = np.zeros((max_node_num, max_node_edge_num), dtype=np.int32)
            for i, num in enumerate(current_node_edge_num):
                node_edge_mask[i, :num] = 1
            batch_node_edge_mask.append(node_edge_mask)

            node_edges = np.full((max_node_num, max_node_edge_num), current_edge_num, dtype=np.int32)
            for i, x in enumerate(graph["node_edges"]):
                node_edges[i, :len(x)] = np.asarray(x, dtype=np.int32)
            batch_node_edges.append(node_edges)

        graph_embed_field = dict()
        graph_embed_field["personal_nodes"] = np.concatenate(
            [[personal_nodes] for personal_nodes in batch_personal_nodes])
        graph_embed_field["feasible_personal_info_nodes"] = np.concatenate(
            [[feasible_personal_info_nodes] for feasible_personal_info_nodes in batch_feasible_personal_info_nodes])
        graph_embed_field["static_feature"] = np.concatenate(
            [[static_feature] for static_feature in batch_static_feature])
        graph_embed_field["edges"] = np.concatenate([[edges] for edges in batch_edges])
        graph_embed_field["node_edge_mask"] = np.concatenate(
            [[node_edge_mask] for node_edge_mask in batch_node_edge_mask])
        graph_embed_field["node_edges"] = np.concatenate([[node_edges] for node_edges in batch_node_edges])

        return graph_embed_field, batch_node_num, max_node_num, max_answer_node_num, max_qa_pair_num, batch_qa_pair_num

    @staticmethod
    def get_knowledge_sampler_field(batch):
        """  Used in User Simulators  """
        batch_knowledge_sampler_filed = list()
        for graph in batch:
            personal_information = copy.deepcopy(graph["personal_information"])
            one_step_node_edges = copy.deepcopy(graph["one_step_node_edges"])
            adj_matrix = np.asarray(graph["adj_matrix"], dtype=bool)
            edge_se_freqs_matrix = np.asarray(graph["edge_se_freqs_matrix"], dtype=np.float32)
            edges = copy.deepcopy(graph["edges"])
            identity_dict_keys = list(graph["one_step_node_edges"].keys())
            idx2node = copy.deepcopy(graph["idx2node"])
            knowledge_sampler_filed = dict()
            knowledge_sampler_filed["personal_information"] = personal_information
            knowledge_sampler_filed["one_step_node_edges"] = one_step_node_edges
            knowledge_sampler_filed["adj_matrix"] = adj_matrix
            knowledge_sampler_filed["edge_se_freqs_matrix"] = edge_se_freqs_matrix
            knowledge_sampler_filed["edges"] = edges
            knowledge_sampler_filed["identity_dict_keys"] = identity_dict_keys
            knowledge_sampler_filed["idx2node"] = idx2node
            knowledge_sampler_filed["identity_dict_values"] = PLACE_HOLDER
            knowledge_sampler_filed["results"] = PLACE_HOLDER
            batch_knowledge_sampler_filed.append(knowledge_sampler_filed)
        return batch_knowledge_sampler_filed

    @staticmethod
    def get_language_generation_field(batch):
        batch_language_generation_filed = list()
        for graph in batch:
            language_generation_filed = dict()
            language_generation_filed["idx2node"] = copy.deepcopy(graph["idx2node"])
            h_t_to_r = dict()
            for edge in graph["edges"]:
                h = edge[2]
                t = edge[0]
                r = edge[1]
                key = str(h) + " " + str(t)
                value = str(r)
                h_t_to_r[key] = value
            language_generation_filed["h_t_to_r"] = h_t_to_r
            batch_language_generation_filed.append(language_generation_filed)
        return batch_language_generation_filed

    def generator(self, data_set_name, shuffle=True):
        if data_set_name == "train":
            data_set = self.train_set
        elif data_set_name == "dev":
            data_set = self.dev_set
        else:
            data_set = self.test_set

        size = len(data_set)
        data_set_idx = list(range(size))
        if shuffle:
            random.shuffle(data_set_idx)

        for idx in data_set_idx:
            yield {"graph_embed_field": data_set[idx]["graph_embed_field"],
                   "policy_field": data_set[idx]["policy_field"],
                   "knowledge_sampler_filed": copy.deepcopy(data_set[idx]["knowledge_sampler_filed"]),
                   "language_generation_filed": data_set[idx]["language_generation_filed"],
                   "state_tracker_field": copy.deepcopy(data_set[idx]["state_tracker_field"])}


class GraphPreprocessHRL(GraphPreprocess):
    def __init__(self, data_set_dir="preprocessed_graphs", batch_size=32):
        super(GraphPreprocessHRL, self).__init__(data_set_dir, batch_size)

    def preprocess_batch(self, batch):
        graph_embed_field, batch_node_num, max_node_num, max_answer_node_num, _, _ = self.get_graph_embed_filed(batch)
        policy_field = self.get_policy_field(batch, batch_node_num, max_answer_node_num)
        knowledge_sampler_filed = self.get_knowledge_sampler_field(batch)
        language_generation_filed = self.get_language_generation_field(batch)
        state_tracker_field = self.get_state_tracker_field(batch, max_node_num, max_answer_node_num)
        preprocessed_batch = {"graph_embed_field": graph_embed_field,
                              "policy_field": policy_field,
                              "knowledge_sampler_filed": knowledge_sampler_filed,
                              "language_generation_filed": language_generation_filed,
                              "state_tracker_field": state_tracker_field}
        return preprocessed_batch

    @staticmethod
    def get_state_tracker_field(batch, max_node_num, max_answer_node_num):
        """  A state tracker for each time step of a dialogue.
             It records current state,
             the mask information,
             the sample information of manager and workers,
             execute an action based on the state and get some reward  """
        batch_state_tracker_field = list()
        for graph in batch:
            # store state information of each time step in a list
            state_tracker_field = list()

            # this is the initial dialogue state
            # for each time step, we generate a same data structure and append it.
            state_in_initial_step = dict()

            # some constant
            # used in generate dialogue feature
            state_in_initial_step["max_node_num"] = max_node_num
            state_in_initial_step["nodes"] = copy.deepcopy(graph["nodes"])
            # used in move a new step
            state_in_initial_step["personal_nodes"] = copy.deepcopy(graph["personal_nodes"])

            # for all nodes
            state_in_initial_step["explored_nodes"] = set()
            state_in_initial_step["last_turn_q_node"] = None
            state_in_initial_step["last_turn_a_node"] = None
            state_in_initial_step["not_explored_nodes"] = set(copy.deepcopy(graph["nodes"]))

            # only for the answer nodes
            state_in_initial_step["known_nodes"] = set()
            state_in_initial_step["unknown_nodes"] = set()
            state_in_initial_step["not_answered_nodes"] = set(
                list(chain(*graph["one_step_nodes"].values())))

            # dialogue feature of current state, calculate before all NN operations
            state_in_initial_step["dialogue_feature"] = PLACE_HOLDER

            # episode end flag
            state_in_initial_step["episode_not_end"] = True

            # exploring turn counter
            state_in_initial_step["total_exploring_turn"] = 0
            state_in_initial_step["current_worker_exploring_turn"] = 0

            # policy mask, mark the running policy of current state
            policy_mask = [1] + [0 for _ in graph["personal_nodes"]]
            state_in_initial_step["policy_mask"] = np.asarray(policy_mask, dtype=np.int32)

            # manager action mask
            # create two counterparts, one for RL, one for RuleWarmUp
            manager_action_mask = [0 for _ in graph["personal_nodes"]]
            for worker_idx, answer_nodes in graph["one_step_nodes"].items():
                if len(answer_nodes) > 0:
                    manager_action_mask[graph["personal_nodes"].index(int(worker_idx))] = 1
            # append [0, 0] for two terminal actions
            manager_action_mask += [0, 0]
            state_in_initial_step["rl_manager_action_mask"] = np.asarray(manager_action_mask, dtype=np.int32)
            state_in_initial_step["warm_up_manager_action_mask"] = np.asarray(manager_action_mask, dtype=np.int32)

            # the workers and manager decision and if they are right
            # used to get the reward Bonus
            state_in_initial_step["workers_decision"] = [PLACE_HOLDER for _ in graph["personal_nodes"]]
            state_in_initial_step["workers_success_state"] = [PLACE_HOLDER for _ in graph["personal_nodes"]]
            state_in_initial_step["manager_decision"] = PLACE_HOLDER
            state_in_initial_step["manager_success_state"] = PLACE_HOLDER

            # worker action mask
            # similar to manager action mask, the terminal action is available only after giving a few questions
            # create two counterparts, one for RL, one for RuleWarmUp
            workers_action_mask = np.zeros((len(graph["personal_nodes"]), max_answer_node_num + 2), dtype=np.int32)
            answer_node_num = list()
            for worker in graph["personal_nodes"]:
                answer_node_num.append(len(graph["one_step_nodes"][str(worker)]))
            valid_worker_action_idx = list()
            row, col = workers_action_mask.shape
            for i, num in enumerate(answer_node_num):
                for idx in range(num):
                    valid_worker_action_idx.append(idx + i * col)
            workers_action_mask = workers_action_mask.reshape(-1)
            workers_action_mask[valid_worker_action_idx] = 1
            workers_action_mask = workers_action_mask.reshape((row, col))
            state_in_initial_step["rl_workers_action_mask"] = workers_action_mask
            state_in_initial_step["warm_up_workers_action_mask"] = workers_action_mask.copy()

            # the manager sample action idx, used for NN update
            state_in_initial_step["manager_sample_idx"] = PLACE_HOLDER

            # the workers sample action idx, used for NN update
            state_in_initial_step["workers_sample_idx"] = PLACE_HOLDER

            # the system action execute in current state
            state_in_initial_step["system_action"] = PLACE_HOLDER

            # reward recorder, include reward mask, manager reward, workers reward
            state_in_initial_step["reward_mask"] = np.copy(state_in_initial_step["policy_mask"])
            state_in_initial_step["manager_reward"] = np.zeros((1,), dtype=np.float32)
            state_in_initial_step["workers_reward"] = np.zeros((len(state_in_initial_step["personal_nodes"]),),
                                                               dtype=np.float32)

            # qa statistical information
            state_in_initial_step["total_qa_turn"] = 0

            # workers qa statistical information
            state_in_initial_step["workers_qa_turn"] = np.zeros((len(graph["personal_nodes"]),), dtype=np.int32)

            # debug
            state_in_initial_step["manager_action_prob"] = PLACE_HOLDER
            state_in_initial_step["workers_action_prob"] = PLACE_HOLDER

            # for Rule based warm up
            state_in_initial_step["workers_counter"] = [{"Known": 0, "UnKnown": 0},
                                                        {"Known": 0, "UnKnown": 0},
                                                        {"Known": 0, "UnKnown": 0},
                                                        {"Known": 0, "UnKnown": 0}]

            # for divide in loss
            state_in_initial_step["valid_workers_num"] = 0

            state_tracker_field.append(state_in_initial_step)
            batch_state_tracker_field.append(state_tracker_field)

        return batch_state_tracker_field

    @staticmethod
    def get_policy_field(batch, batch_node_num, max_answer_node_num):
        batch_manager_actions = list()
        batch_workers_actions = list()
        batch_workers_max_qa_turn = list()

        for graph, current_node_num in zip(batch, batch_node_num):
            manager_actions = np.asarray(graph["personal_nodes"], dtype=np.int32)
            batch_manager_actions.append(manager_actions)

            workers_actions = np.full((len(graph["personal_nodes"]), max_answer_node_num), current_node_num,
                                      dtype=np.int32)
            workers_max_qa_turn = np.zeros((len(graph["personal_nodes"]),), dtype=np.int32)
            for worker_idx, answer_nodes in graph["one_step_nodes"].items():
                if len(answer_nodes) > 0:
                    workers_actions[int(worker_idx), :len(answer_nodes)] = np.asarray(answer_nodes, dtype=np.int32)
                    workers_max_qa_turn[int(worker_idx)] = len(answer_nodes) if len(
                        answer_nodes) < MaxWorkerExploringTimeStep else MaxWorkerExploringTimeStep

            batch_workers_actions.append(workers_actions)
            batch_workers_max_qa_turn.append(workers_max_qa_turn)

        policy_field = dict()
        policy_field["manager_actions"] = np.concatenate(
            [[manager_actions] for manager_actions in batch_manager_actions])
        policy_field["workers_actions"] = np.concatenate(
            [[workers_actions] for workers_actions in batch_workers_actions])
        policy_field["workers_max_qa_turn"] = np.concatenate(
            [[workers_max_qa_turn] for workers_max_qa_turn in batch_workers_max_qa_turn])

        return policy_field


class GraphPreprocessRL(GraphPreprocess):
    def __init__(self, data_set_dir="preprocessed_graphs", batch_size=32):
        super(GraphPreprocessRL, self).__init__(data_set_dir, batch_size)

    def preprocess_batch(self, batch):
        graph_embed_field, batch_node_num, max_node_num, _, max_qa_pair_num, batch_qa_pair_num = self.get_graph_embed_filed(
            batch)
        policy_field = self.get_policy_field(batch, batch_node_num, max_qa_pair_num)
        knowledge_sampler_filed = self.get_knowledge_sampler_field(batch)
        language_generation_filed = self.get_language_generation_field(batch)
        state_tracker_field = self.get_state_tracker_field(batch, max_node_num, max_qa_pair_num, batch_qa_pair_num)
        preprocessed_batch = {"graph_embed_field": graph_embed_field,
                              "policy_field": policy_field,
                              "knowledge_sampler_filed": knowledge_sampler_filed,
                              "language_generation_filed": language_generation_filed,
                              "state_tracker_field": state_tracker_field}
        return preprocessed_batch

    @staticmethod
    def get_policy_field(batch, batch_node_num, max_qa_pair_num):
        batch_actions = list()
        batch_workers_max_qa_turn = list()

        for graph, current_node_num in zip(batch, batch_node_num):
            actions = list()
            for q_node, answer_nodes in graph["one_step_nodes"].items():
                for answer_node in answer_nodes:
                    actions.append((int(q_node), int(answer_node)))
            for _ in range(max_qa_pair_num - len(actions)):
                actions.append((int(current_node_num), int(current_node_num)))
            batch_actions.append(actions)

            workers_max_qa_turn = np.zeros((len(graph["personal_nodes"]),), dtype=np.int32)
            batch_workers_max_qa_turn.append(workers_max_qa_turn)

        policy_field = dict()
        policy_field["actions"] = np.concatenate(
            [[actions] for actions in batch_actions])
        policy_field["workers_max_qa_turn"] = np.concatenate(
            [[workers_max_qa_turn] for workers_max_qa_turn in batch_workers_max_qa_turn])

        return policy_field

    @staticmethod
    def get_state_tracker_field(batch, max_node_num, max_qa_pair_num, batch_qa_pair_num):
        batch_state_tracker_field = list()
        for graph, qa_pair_num in zip(batch, batch_qa_pair_num):
            state_tracker_field = list()
            state_in_initial_step = dict()

            state_in_initial_step["max_node_num"] = max_node_num
            state_in_initial_step["nodes"] = copy.deepcopy(graph["nodes"])

            # for all nodes
            state_in_initial_step["explored_nodes"] = set()
            state_in_initial_step["last_turn_q_node"] = None
            state_in_initial_step["last_turn_a_node"] = None
            state_in_initial_step["not_explored_nodes"] = set(copy.deepcopy(graph["nodes"]))

            # only for the answer nodes
            state_in_initial_step["known_nodes"] = set()
            state_in_initial_step["unknown_nodes"] = set()
            state_in_initial_step["not_answered_nodes"] = set(
                list(chain(*graph["one_step_nodes"].values())))

            # dialogue feature of current state, calculate before all NN operations
            state_in_initial_step["dialogue_feature"] = PLACE_HOLDER

            # episode end flag
            state_in_initial_step["episode_not_end"] = True

            # exploring turn counter
            state_in_initial_step["total_exploring_turn"] = 0

            # rl action mask
            rl_action_mask = np.zeros((max_qa_pair_num + 2,), dtype=np.int32)
            rl_action_mask[:qa_pair_num] = 1
            state_in_initial_step["rl_action_mask"] = rl_action_mask
            state_in_initial_step["warm_up_action_mask"] = rl_action_mask.copy()

            state_in_initial_step["decision"] = PLACE_HOLDER
            state_in_initial_step["success_state"] = PLACE_HOLDER

            state_in_initial_step["sample_idx"] = PLACE_HOLDER

            state_in_initial_step["system_action"] = PLACE_HOLDER

            state_in_initial_step["reward"] = np.zeros((), dtype=np.float32)

            state_in_initial_step["total_qa_turn"] = 0

            # debug
            state_in_initial_step["action_prob"] = PLACE_HOLDER

            # for Rule based warm up
            state_in_initial_step["workers_counter"] = [{"Known": 0, "UnKnown": 0},
                                                        {"Known": 0, "UnKnown": 0},
                                                        {"Known": 0, "UnKnown": 0},
                                                        {"Known": 0, "UnKnown": 0}]

            state_tracker_field.append(state_in_initial_step)
            batch_state_tracker_field.append(state_tracker_field)

        return batch_state_tracker_field
