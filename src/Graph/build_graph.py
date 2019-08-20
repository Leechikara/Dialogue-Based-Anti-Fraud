# coding = utf-8
"""
Build user information Knowledge Graph.
Transfer the original Knowledge Graph into idx format.
Get the static feature of nodes in the KG.
Split the data set into train set, dev set and test set.
"""
import json
import os
import locale
import random
import numpy as np
from itertools import chain
from collections import OrderedDict
from src.config import DATA_ROOT, train_size, dev_size, test_size
from src.Graph.node_feature import NodeFeature

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


class Graph(object):
    def __init__(self, user_id,
                 complete_kg_dir="complete_kg",
                 se_result_dir="se_result"):
        self.user_id = user_id
        self.nodes = None
        self.personal_nodes = None
        self.personal_information = OrderedDict.fromkeys(["company", "university", "live_in", "born_in"])
        self.one_step_nodes = None
        self.node2idx = None
        self.idx2node = None
        self.edges = None
        self.edge_se_freqs = None
        self.graph = None
        self.transferred_graph = None

        self.load_edges(complete_kg_dir, se_result_dir)
        self.load_nodes(complete_kg_dir)
        self.generate_graph()

    def load_edges(self, complete_kg_dir, se_result_dir):
        with open(os.path.join(DATA_ROOT, complete_kg_dir, str(self.user_id) + ".json"), "r") as f:
            complete_kg = json.load(f)
        with open(os.path.join(DATA_ROOT, se_result_dir, str(self.user_id) + ".json"), "r") as f:
            se_result = json.load(f)

        self.edges = list()
        self.edge_se_freqs = list()

        for triples in complete_kg["paths"]["PersonEntity"].values():
            for (h, r, t) in triples:
                # reverse the triple
                self.edges.append((t, r, h))
                se_query = "“{}” “{}”".format(str(h), str(t))
                if se_query not in se_result.keys():
                    se_query = "“{}” “{}”".format(str(t), str(h))
                assert se_query in se_result.keys()
                self.edge_se_freqs.append(locale.atoi(se_result[se_query]))

        for (h, r, t) in complete_kg["paths"]["NewAdd"]:
            self.edges.append((t, r, h))
            se_query = "“{}” “{}”".format(str(h), str(t))
            if se_query not in se_result.keys():
                se_query = "“{}” “{}”".format(str(t), str(h))
            assert se_query in se_result.keys()
            self.edge_se_freqs.append(locale.atoi(se_result[se_query]))

    def load_nodes(self, complete_kg_dir):
        with open(os.path.join(DATA_ROOT, complete_kg_dir, str(self.user_id) + ".json"), "r") as f:
            complete_kg = json.load(f)

        self.nodes = list()
        self.one_step_nodes = dict()

        for (_, _, t) in complete_kg["paths"]["User"]:
            if t not in self.nodes:
                self.nodes.append(t)
        for personal_node, paths in complete_kg["paths"]["PersonEntity"].items():
            assert personal_node not in self.one_step_nodes.keys()
            self.one_step_nodes[personal_node] = list()
            for (h, r, t) in paths:
                if h not in self.nodes:
                    self.nodes.append(h)
                if t not in self.nodes:
                    self.nodes.append(t)
                if t not in self.one_step_nodes[personal_node]:
                    self.one_step_nodes[personal_node].append(t)
        for (h, r, t) in complete_kg["paths"]["NewAdd"]:
            for node in self.one_step_nodes.keys():
                if h == node and t not in self.one_step_nodes[node]:
                    self.one_step_nodes[node].append(t)

        self.node2idx = dict()
        self.idx2node = dict()
        for i, node in enumerate(self.nodes):
            self.node2idx[node] = i
            self.idx2node[i] = node

        self.personal_nodes = list()
        for (_, r, t) in complete_kg["paths"]["User"]:
            self.personal_nodes.append(self.node2idx[t])
            if r == "曾工作于":
                self.personal_information["company"] = self.node2idx[t]
            elif r == "毕业于":
                self.personal_information["university"] = self.node2idx[t]
            elif r == "住在":
                self.personal_information["live_in"] = self.node2idx[t]
            elif r == "出生于":
                self.personal_information["born_in"] = self.node2idx[t]
            else:
                raise ValueError('Unknown personal information type.')

    def generate_graph(self):
        self.graph = dict()
        self.graph["nodes"] = self.nodes
        self.graph["personal_nodes"] = self.personal_nodes
        self.graph["personal_information"] = self.personal_information
        self.graph["node2idx"] = self.node2idx
        self.graph["idx2node"] = self.idx2node
        self.graph["edges"] = self.edges
        self.graph["edge_se_freqs"] = self.edge_se_freqs
        self.graph["one_step_nodes"] = self.one_step_nodes

        node_edges = dict()
        for node in self.nodes:
            edges = list()
            for edge in self.edges:
                if node == edge[2]:
                    edges.append(edge)
            node_edges[node] = edges
        self.graph["node_edges"] = node_edges

        one_step_node_edges = dict()
        for node in self.one_step_nodes.keys():
            assert node not in one_step_node_edges.keys()
            edges = list()
            for edge in self.edges:
                if edge[0] in self.one_step_nodes[node] or edge[2] in self.one_step_nodes[node]:
                    edges.append(edge)
            one_step_node_edges[node] = edges
        self.graph["one_step_node_edges"] = one_step_node_edges
        assert set(list(chain(*self.graph["one_step_node_edges"].values()))) == set(self.graph["edges"])

    def all_relations(self, relation_list):
        edges = self.graph["edges"]
        for h, r, t in edges:
            if r not in relation_list:
                relation_list.append(r)
        return relation_list

    def build_answer_library(self, answer_library):
        edges = self.graph["edges"]
        for h, r, t in edges:
            if r not in answer_library:
                answer_library[r] = {h}
            else:
                answer_library[r].add(h)
        return answer_library

    @staticmethod
    def edge2idx(edge, edges):
        for idx, (h, r, t) in enumerate(edges):
            if edge[0] == h and edge[1] == r and edge[2] == t:
                return idx

    def transfer_graph_to_idx(self, r2idx):
        self.transferred_graph = dict()

        self.transferred_graph["nodes"] = list(range(len(self.graph["nodes"])))

        self.transferred_graph["personal_nodes"] = self.graph["personal_nodes"]

        self.transferred_graph["personal_information"] = self.graph["personal_information"]

        self.transferred_graph["one_step_nodes"] = dict()
        for key, values in self.graph["one_step_nodes"].items():
            key_idx = self.graph["node2idx"][key]
            self.transferred_graph["one_step_nodes"][key_idx] = [self.graph["node2idx"][v] for v in values]

        self.transferred_graph["one_step_node_edges"] = dict()
        for node, edges in self.graph["one_step_node_edges"].items():
            node_idx = self.graph["node2idx"][node]
            self.transferred_graph["one_step_node_edges"][node_idx] = list()
            for _h, _r, _t in edges:
                h = self.graph["node2idx"][_h]
                t = self.graph["node2idx"][_t]
                r = r2idx[_r]
                self.transferred_graph["one_step_node_edges"][node_idx].append((h, r, t))

        self.transferred_graph["edges"] = list()
        for _h, _r, _t in self.graph["edges"]:
            h = self.graph["node2idx"][_h]
            t = self.graph["node2idx"][_t]
            r = r2idx[_r]
            self.transferred_graph["edges"].append((h, r, t))

        self.transferred_graph["node_edges"] = list()
        for _ in self.transferred_graph["nodes"]:
            self.transferred_graph["node_edges"].append(list())
        for node, edges in self.graph["node_edges"].items():
            node_idx = self.graph["node2idx"][node]
            for edge in edges:
                edge_idx = self.edge2idx(edge, self.graph["edges"])
                self.transferred_graph["node_edges"][node_idx].append(edge_idx)

        self.transferred_graph["idx2node"] = self.graph["idx2node"]

        self.transferred_graph["node_degree"] = list()
        self.transferred_graph["node_in_degree"] = list()
        self.transferred_graph["node_out_degree"] = list()
        for node_idx in self.transferred_graph["nodes"]:
            in_degree = 0
            out_degree = 0
            for (h, r, t) in self.transferred_graph["edges"]:
                if node_idx == h:
                    out_degree += 1
                elif node_idx == t:
                    in_degree += 1
            self.transferred_graph["node_degree"].append(in_degree + out_degree)
            self.transferred_graph["node_in_degree"].append(in_degree)
            self.transferred_graph["node_out_degree"].append(out_degree)

        # valid se freqs only for answer nodes.
        # the se freqs are se results number of triples (personal node, answer node).
        self.transferred_graph["node_se_freqs"] = list()
        for node_idx in self.transferred_graph["nodes"]:
            if node_idx not in list(chain(*self.transferred_graph["one_step_nodes"].values())):
                self.transferred_graph["node_se_freqs"].append(0)
            else:
                max_se_freq = 0
                for se_freq, (h, r, t) in zip(self.graph["edge_se_freqs"], self.transferred_graph["edges"]):
                    if h == node_idx and t in self.transferred_graph["personal_nodes"] and max_se_freq < se_freq:
                        max_se_freq = se_freq
                self.transferred_graph["node_se_freqs"].append(max_se_freq)

        node_num = len(self.graph["nodes"])
        self.transferred_graph["adj_matrix"] = np.eye(node_num, dtype=bool)
        for (h, r, t) in self.transferred_graph["edges"]:
            self.transferred_graph["adj_matrix"][h][t] = True
            self.transferred_graph["adj_matrix"][t][h] = True
        self.transferred_graph["adj_matrix"] = self.transferred_graph["adj_matrix"].tolist()

        self.transferred_graph["edge_se_freqs_matrix"] = np.full((node_num, node_num), float("-inf"))
        np.fill_diagonal(self.transferred_graph["edge_se_freqs_matrix"], float("inf"))
        for (h, r, t), se_freqs in zip(self.transferred_graph["edges"], self.graph["edge_se_freqs"]):
            self.transferred_graph["edge_se_freqs_matrix"][h][t] = se_freqs
        self.transferred_graph["edge_se_freqs_matrix"] = self.transferred_graph["edge_se_freqs_matrix"].tolist()

    def get_graph_static_feature(self, node_feature_generator):
        nodes = self.transferred_graph["nodes"]
        personal_information = self.transferred_graph["personal_information"]
        one_step_nodes = self.transferred_graph["one_step_nodes"]
        node_se_freqs = self.transferred_graph["node_se_freqs"]
        node_degree = self.transferred_graph["node_degree"]
        node_in_degree = self.transferred_graph["node_in_degree"]
        node_out_degree = self.transferred_graph["node_out_degree"]
        static_feature = node_feature_generator.static_feature(nodes, personal_information, one_step_nodes,
                                                               node_se_freqs, node_degree,
                                                               node_in_degree, node_out_degree)
        self.transferred_graph["static_feature"] = static_feature


def main():
    random.seed(0)
    user_list = list(range(1, 907))

    original_graphs_dir = os.path.join(DATA_ROOT, "original_graphs")
    if not os.path.exists(original_graphs_dir):
        os.mkdir(original_graphs_dir)

    graphs = dict()
    all_relations = list()
    answer_library = dict()
    for user_idx in user_list:
        graph = Graph(user_idx)
        graphs[user_idx] = graph
        all_relations = graph.all_relations(all_relations)
        answer_library = graph.build_answer_library(answer_library)
        with open(os.path.join(original_graphs_dir, str(user_idx) + ".json"), "w") as f:
            json.dump(graph.graph, f, ensure_ascii=False, indent=2)

    for k, v in answer_library.items():
        answer_library[k] = list(v)
    with open(os.path.join(DATA_ROOT, "answersLibrary.json"), "w") as f:
        json.dump(answer_library, f, ensure_ascii=False, indent=2)

    r2idx = dict()
    idx2r = dict()
    for i, r in enumerate(all_relations):
        r2idx[r] = i + 1
        idx2r[i + 1] = r
    with open(os.path.join(DATA_ROOT, "r2idx.json"), "w") as f:
        json.dump(r2idx, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DATA_ROOT, "idx2r.json"), "w") as f:
        json.dump(idx2r, f, ensure_ascii=False, indent=2)

    for user_idx in user_list:
        graphs[user_idx].transfer_graph_to_idx(r2idx)

    node_feature_generator = NodeFeature()
    for user_idx in user_list:
        graphs[user_idx].get_graph_static_feature(node_feature_generator)

    # split train dev test
    random.shuffle(user_list)

    train_set = list()
    dev_set = list()
    test_set = list()

    for i in range(train_size):
        train_set.append(graphs[user_list[i]].transferred_graph)

    for i in range(dev_size):
        dev_set.append(graphs[user_list[i + train_size]].transferred_graph)

    for i in range(test_size):
        test_set.append(graphs[user_list[i + train_size + dev_size]].transferred_graph)

    preprocessed_graphs_dir = os.path.join(DATA_ROOT, "preprocessed_graphs")
    if not os.path.exists(preprocessed_graphs_dir):
        os.mkdir(preprocessed_graphs_dir)

    with open(os.path.join(preprocessed_graphs_dir, "train.json"), "w") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)
    with open(os.path.join(preprocessed_graphs_dir, "dev.json"), "w") as f:
        json.dump(dev_set, f, ensure_ascii=False, indent=2)
    with open(os.path.join(preprocessed_graphs_dir, "test.json"), "w") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
