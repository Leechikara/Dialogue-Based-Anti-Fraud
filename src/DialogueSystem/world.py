# coding = utf-8
import numpy as np
import random
import torch.optim as optim
import torch
import time
import os
import json
from src.DialogueSystem.rollout import rollout_hierarchy_policy, rollout_flatten_policy
from src.DialogueSystem.train import train_hierarchy_policy, train_flatten_policy
from src.Graph.preprocess_graph import GraphPreprocessHRL, GraphPreprocessRL
from src.User.user_simulator import UserSimulator
from src.StateTracker.state_traker import StateTrackerHRL, StateTrackerRL
from src.DialogueRecorder.dialogue_recorder import DialogueRecorder
from src.NNModule.graph_neural_network import GNN, ManagerStateTracker, WorkersStateTracker
from src.NNModule.agents import Manager, Workers, Agent
from src.NLG.language_generation import LanguageGenerator
from src.config import device, Init_Node_Feature_Size, ManagerStateRest, WorkersStateRest, DATA_ROOT


class World(object):
    def __init__(self, args):
        self.args = args
        self.set_random_seed()
        self.models = self.build_init_models()
        self.graph_preprocess = self.build_graph_preprocess()
        if self.args.trained_model is not None:
            self.load_pre_trained_models()
        self.optimizer = self.build_optimizer()
        self.Rollout = rollout_hierarchy_policy if self.args.use_hierarchy_policy else rollout_flatten_policy
        self.Train = train_hierarchy_policy if self.args.use_hierarchy_policy else train_flatten_policy
        self.StateTracker = StateTrackerHRL if self.args.use_hierarchy_policy else StateTrackerRL

    def set_random_seed(self):
        random.seed(self.args.rand_seed)
        np.random.seed(self.args.rand_seed + 1)
        torch.manual_seed(self.args.rand_seed + 2)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.rand_seed + 3)

    def build_init_models(self):
        if self.args.new_node_emb_size_list is not None:
            node_emb_size_list = [Init_Node_Feature_Size] + self.args.new_node_emb_size_list
        else:
            node_emb_size_list = [Init_Node_Feature_Size]
        graph_node_emb_size = sum(node_emb_size_list)
        manager_state_size = self.args.global_agg_size + ManagerStateRest
        models = dict()
        models["gnn"] = GNN(node_emb_size_list, self.args.msg_agg).to(device)
        if self.args.use_hierarchy_policy:
            worker_state_size = graph_node_emb_size + WorkersStateRest
            models["manager_state_tracker"] = ManagerStateTracker(graph_node_emb_size, self.args.global_agg_size,
                                                                  self.args.msg_agg).to(device)
            models["workers_state_tracker"] = WorkersStateTracker().to(device)
            models["manager"] = Manager(self.args.score_method, manager_state_size, worker_state_size).to(device)
            models["workers"] = Workers(self.args.score_method, worker_state_size, graph_node_emb_size).to(device)
        else:
            models["manager_state_tracker"] = ManagerStateTracker(graph_node_emb_size, self.args.global_agg_size,
                                                                  self.args.msg_agg).to(device)
            models["manager"] = Agent(self.args.score_method, manager_state_size, graph_node_emb_size).to(device)
        return models

    def build_optimizer(self):
        params = list()
        for model in self.models.values():
            params.extend(model.parameters())
        optimizer = optim.Adam(params, lr=3e-4, weight_decay=0.01)
        return optimizer

    def set_learning_rate(self, learning_rate=3e-4):
        for g in self.optimizer.param_groups:
            g["lr"] = learning_rate

    def build_graph_preprocess(self):
        if self.args.use_hierarchy_policy:
            return GraphPreprocessHRL(batch_size=self.args.batch_size)
        else:
            return GraphPreprocessRL(batch_size=self.args.batch_size)

    def load_pre_trained_models(self):
        path = os.path.join(DATA_ROOT, "checkpoints", self.args.model_setting, self.args.trained_model)
        print("Load trained models in {}.".format(path))

        for model_name in self.models.keys():
            model_path = os.path.join(path, model_name + ".pkl")
            self.models[model_name].load_state_dict(torch.load(model_path))

    def save_models(self, log):
        path = os.path.join(DATA_ROOT, "checkpoints", self.args.model_setting)
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, log["train_mode"])
        if not os.path.exists(path):
            os.mkdir(path)
        current_state = "_success_" + str(log["success"]) + "_turn_" + str(log["turn"])
        path = os.path.join(path, "epoch_" + str(log["epoch"]) + current_state)
        if not os.path.exists(path):
            os.mkdir(path)
        print("Save the models in {}.".format(path))
        for model_name in self.models.keys():
            torch.save(self.models[model_name].state_dict(), os.path.join(path, model_name + ".pkl"))

    def save_running_record(self, log):
        path = os.path.join(DATA_ROOT, "checkpoints", self.args.model_setting)
        if not os.path.exists(path):
            os.mkdir(path)
        with open(os.path.join(path, "running_record.json"), "w") as f:
            json.dump(log, f, indent=2)

    def warm_up(self):
        self.set_learning_rate(self.args.warm_up_learning_rate)
        training_logs = {"best": {"success": 0, "checkpoint": None},
                         "running_record": {}}
        sample_flag = "random"
        mode = "RuleWarmUp"

        for epoch in range(1, self.args.warm_up_epoch + 1):
            t1 = time.time()

            for batch in self.graph_preprocess.generator("train", shuffle=True):
                graph_embed_field = batch["graph_embed_field"]
                policy_field = batch["policy_field"]
                users = [UserSimulator(filed) for filed in batch["knowledge_sampler_filed"]]
                state_tracker_field = batch["state_tracker_field"]
                state_trackers = [self.StateTracker(field) for field in state_tracker_field]
                language_generators = [LanguageGenerator(field) for field in batch["language_generation_filed"]]
                dialogue_recorders = [None for _ in range(self.args.batch_size)]

                with torch.no_grad():
                    self.Rollout(graph_embed_field, policy_field, state_tracker_field,
                                 state_trackers, users, language_generators, dialogue_recorders, sample_flag,
                                 **self.models)

                with torch.enable_grad():
                    loss = self.Train(graph_embed_field, policy_field, state_tracker_field,
                                      self.optimizer, self.args.max_clip, self.args.entropy_coef, mode,
                                      **self.models)
                if self.args.verbose:
                    print("Rule based warm up loss in current batch: {}.".format(loss))

            t2 = time.time()
            print("Run warm up, cost {:.1f}s in this epoch.".format(t2 - t1))

            # evaluate and save models
            if epoch % self.args.eval_interval == 0:
                average_success, average_turn = self.evaluate()
                print("After {} epochs warm up, the average success rate in dev set is {}.".format(epoch,
                                                                                                   average_success))
                print("After {} epochs warm up, the average turns in dev set is {}.".format(epoch, average_turn))
                if average_success > training_logs["best"]["success"]:
                    checkpoint = "epoch_" + str(epoch) + "_success_" + str(average_success) + "_turn_" + str(
                        average_turn)
                    training_logs["best"] = {"success": average_success,
                                             "checkpoint": checkpoint}
                training_logs["running_record"][epoch] = {"success": average_success, "turn": average_turn}
                logs = {"epoch": epoch, "success": average_success, "turn": average_turn, "train_mode": "RuleWarmUp"}
                self.save_models(logs)

        return training_logs

    def rl(self):
        self.set_learning_rate(self.args.rl_learning_rate)
        training_logs = {"best": {"success": 0, "checkpoint": None},
                         "running_record": {}}

        for epoch in range(1, self.args.rl_epoch + 1):
            t1 = time.time()

            for batch_id, batch in enumerate(self.graph_preprocess.generator("train", shuffle=True)):
                graph_embed_field = batch["graph_embed_field"]
                policy_field = batch["policy_field"]
                users = [UserSimulator(filed) for filed in batch["knowledge_sampler_filed"]]
                state_tracker_field = batch["state_tracker_field"]
                state_trackers = [self.StateTracker(field) for field in state_tracker_field]
                language_generators = [LanguageGenerator(field) for field in batch["language_generation_filed"]]
                dialogue_recorders = [None for _ in range(self.graph_preprocess.batch_size)]

                if self.args.warm_up and (batch_id + 1) % self.args.warm_up_interval == 0:
                    sample_flag = "random"
                    mode = "RuleWarmUp"
                else:
                    sample_flag = "sample"
                    mode = "RL"

                with torch.no_grad():
                    self.Rollout(graph_embed_field, policy_field, state_tracker_field,
                                 state_trackers, users, language_generators, dialogue_recorders, sample_flag,
                                 **self.models)

                with torch.enable_grad():
                    loss = self.Train(graph_embed_field, policy_field, state_tracker_field,
                                      self.optimizer, self.args.max_clip, self.args.entropy_coef, mode,
                                      **self.models)

                if self.args.verbose:
                    if mode == "RuleWarmUp":
                        print("Rule based warm up loss in current batch: {}.".format(loss))
                    else:
                        print("RL loss in current batch: {}.".format(loss))

            t2 = time.time()
            print("Run RL, cost {:.1f}s in this epoch.".format(t2 - t1))

            # evaluate and save models
            if epoch % self.args.eval_interval == 0:
                average_success, average_turn = self.evaluate()
                print("After {} epochs RL, the average success rate in dev set is {}.".format(epoch, average_success))
                print("After {} epochs RL, the average turns in dev set is {}.".format(epoch, average_turn))
                if average_success > training_logs["best"]["success"]:
                    checkpoint = "epoch_" + str(epoch) + "_success_" + str(average_success) + "_turn_" + str(
                        average_turn)
                    training_logs["best"] = {"success": average_success,
                                             "checkpoint": checkpoint}
                training_logs["running_record"][epoch] = {"success": average_success, "turn": average_turn}
                logs = {"epoch": epoch, "success": average_success, "turn": average_turn, "train_mode": "RL"}
                self.save_models(logs)

        return training_logs

    def evaluate(self):
        sample_flag = "random" if self.args.test_rule_based_system else "max"
        success_list = list()
        turn_list = list()

        for _ in range(self.args.eval_epoch):
            for batch in self.graph_preprocess.generator("test" if self.args.test else "dev", shuffle=False):
                graph_embed_field = batch["graph_embed_field"]
                policy_field = batch["policy_field"]
                users = [UserSimulator(filed) for filed in batch["knowledge_sampler_filed"]]
                state_tracker_field = batch["state_tracker_field"]
                state_trackers = [self.StateTracker(field) for field in state_tracker_field]
                language_generators = [LanguageGenerator(field) for field in batch["language_generation_filed"]]
                if self.args.test and self.args.record_dialogue:
                    dialogue_recorders = [DialogueRecorder() for _ in range(self.args.batch_size)]
                else:
                    dialogue_recorders = [None for _ in range(self.args.batch_size)]

                with torch.no_grad():
                    self.Rollout(graph_embed_field, policy_field, state_tracker_field,
                                 state_trackers, users, language_generators, dialogue_recorders, sample_flag,
                                 **self.models)

                if self.args.use_hierarchy_policy:
                    success_list += [int(filed[-1]["manager_success_state"]) for filed in state_tracker_field]
                else:
                    success_list += [int(filed[-1]["success_state"]) for filed in state_tracker_field]
                turn_list += [int(filed[-1]["total_qa_turn"]) for filed in state_tracker_field]

        return np.mean(np.asarray(success_list)), np.mean(np.asarray(turn_list))

    def run(self):
        if self.args.train:
            training_log = dict()
            if self.args.warm_up:
                print("Start warm up...")
                warm_up_log = self.warm_up()
                training_log["RuleWarmUp"] = warm_up_log

            if self.args.rl:
                print("Start RL...")
                rl_log = self.rl()
                training_log["RL"] = rl_log

            self.save_running_record(training_log)

        if self.args.test:
            print("Start test the model...")
            if self.args.test_rule_based_system:
                print("Test the rule-based system.")
            else:
                print("Test the data-driven system.")
            average_success, average_turn = self.evaluate()
            print("The average success rate in test set is {}.".format(average_success))
            print("The average turns in test set is {}.".format(average_turn))
