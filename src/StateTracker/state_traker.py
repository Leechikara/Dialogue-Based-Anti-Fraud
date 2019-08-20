# coding = utf-8
import copy
import numpy as np
from src.config import NON_FRAUD, FRAUD, Known, UnKnown, PLACE_HOLDER, MaxExploringTimeStep, \
    MaxWorkerExploringTimeStep, ManagerRecognitionCorrect, ManagerRecognitionWrong, WorkerRecognitionCorrect, \
    WorkerRecognitionWrong, ExploringPunish, MinDifference, MinWorkerQATurn, WorkerBonus, ManagerBonus, \
    MaxFlattenRuleQATurn, MinFlattenRLQATurn
from src.Graph.node_feature import NodeFeature


class StateTracker(object):
    def __init__(self, state_tracker_field):
        self.init_episode(state_tracker_field)

    def generate_recent_dialogue_feature(self):
        max_node_num = self.state_tracker_field[-1]["max_node_num"]
        nodes = self.state_tracker_field[-1]["nodes"]
        explored_nodes = self.state_tracker_field[-1]["explored_nodes"]
        last_turn_q_node = self.state_tracker_field[-1]["last_turn_q_node"]
        last_turn_a_node = self.state_tracker_field[-1]["last_turn_a_node"]
        not_explored_nodes = self.state_tracker_field[-1]["not_explored_nodes"]
        known_nodes = self.state_tracker_field[-1]["known_nodes"]
        unknown_nodes = self.state_tracker_field[-1]["unknown_nodes"]
        not_answered_nodes = self.state_tracker_field[-1]["not_answered_nodes"]
        self.state_tracker_field[-1]["dialogue_feature"] = self.node_feat_generator.dialogue_feature(max_node_num,
                                                                                                     nodes,
                                                                                                     explored_nodes,
                                                                                                     last_turn_q_node,
                                                                                                     last_turn_a_node,
                                                                                                     not_explored_nodes,
                                                                                                     known_nodes,
                                                                                                     unknown_nodes,
                                                                                                     not_answered_nodes)

    def init_episode(self, state_tracker_field):
        self.state_tracker_field = state_tracker_field
        self.node_feat_generator = NodeFeature()
        self.generate_recent_dialogue_feature()


class StateTrackerRL(StateTracker):
    def __init__(self, state_tracker_field):
        super(StateTrackerRL, self).__init__(state_tracker_field)

    def generate_system_action(self, sample_idx, sample_content, user_identity_state):
        """ The sample content is a tuple (q node, a node) or (Pad_Query_Node, terminal action). """
        if sample_content[1] in [FRAUD, NON_FRAUD]:
            system_action = {"manager": {"sample_idx": sample_idx,
                                         "sample_content": sample_content[1]}}

            # record agent's decision and its state
            self.state_tracker_field[-1]["decision"] = sample_content[1]
            if sample_content[1] == user_identity_state:
                self.state_tracker_field[-1]["success_state"] = True
            else:
                self.state_tracker_field[-1]["success_state"] = False
        else:
            system_action = {"worker": {"sample_idx": sample_idx,
                                        "sample_content": sample_content[1]}}

        self.state_tracker_field[-1]["system_action"] = system_action
        return system_action

    def generate_reward(self):
        success_state = self.state_tracker_field[-1]["success_state"]

        if success_state is True:
            self.state_tracker_field[-1]["reward"] += ManagerRecognitionCorrect
        else:
            self.state_tracker_field[-1]["reward"] += ManagerRecognitionWrong

        for i in range(len(self.state_tracker_field) - 1):
            self.state_tracker_field[i]["reward"] += ExploringPunish

    def move_a_step(self,
                    language_generator,
                    user,
                    dialogue_recorder,
                    sample_idx,
                    sample_content,
                    action_prob,
                    mode):
        # just for debug
        self.state_tracker_field[-1]["action_prob"] = action_prob

        # pad to the longest episode
        self.state_tracker_field[-1]["sample_idx"] = sample_idx

        if self.state_tracker_field[-1]["episode_not_end"] is False:
            # pad to the longest episode
            self.state_tracker_field.append(copy.deepcopy(self.state_tracker_field[-1]))
        else:
            # generate the system action in the current valid dialogue state first
            system_action = self.generate_system_action(sample_idx, sample_content, user.user_identity_state)
            # and then inherit information of it
            state_in_new_step = copy.deepcopy(self.state_tracker_field[-1])

            state_in_new_step["total_exploring_turn"] += 1
            if state_in_new_step["total_exploring_turn"] >= MaxExploringTimeStep:
                # terminal rl in the next step by force
                state_in_new_step["rl_action_mask"][:-2] = 0
                state_in_new_step["rl_action_mask"][-2:] = 1

            # generate question
            if sample_content[1] not in [FRAUD, NON_FRAUD]:
                q_node, a_node = sample_content[0], sample_content[1]
            else:
                q_node, a_node = None, None
            # user give answer
            user_answer, user_answer_state = user.answer(q_node, a_node)

            # record dialogue
            if dialogue_recorder is not None:
                if user_answer is not None:
                    dialogue_recorder.record_turn(language_generator, q_node=q_node, a_node=a_node,
                                                  user_answer=user_answer)
                elif "manager" in system_action.keys() and system_action["manager"]["sample_content"] in [FRAUD,
                                                                                                          NON_FRAUD]:
                    dialogue_recorder.record_turn(language_generator,
                                                  decision=system_action["manager"]["sample_content"])

            if user_answer is not None:
                state_in_new_step["total_qa_turn"] += 1

            # some information needs to be filled
            # we delete the useless time step in rollout
            state_in_new_step["dialogue_feature"] = PLACE_HOLDER
            state_in_new_step["sample_idx"] = PLACE_HOLDER
            state_in_new_step["system_action"] = PLACE_HOLDER
            state_in_new_step["reward"] = np.zeros((), dtype=np.float32)

            # manager and work are the HRL agents
            # the manager_action_content and worker_action_content correspond to node id in graph or terminal action id
            # the manager_action_idx and worker_action_idx correspond to action idx in logits
            manager_action = system_action.get("manager", {"sample_idx": None, "sample_content": None})
            worker_action = system_action.get("worker", {"sample_idx": None, "sample_content": None})

            manager_action_content = manager_action["sample_content"]
            worker_action_idx = worker_action["sample_idx"]
            worker_action_content = worker_action["sample_content"]

            if manager_action_content in [FRAUD, NON_FRAUD]:
                # terminal reward for manager
                self.generate_reward()

                # end this episode
                state_in_new_step["episode_not_end"] = False
            else:
                # generate for dialogue feature
                state_in_new_step["explored_nodes"].add(worker_action_content)
                if worker_action_content in state_in_new_step["not_explored_nodes"]:
                    state_in_new_step["not_explored_nodes"].remove(worker_action_content)
                if worker_action_content in state_in_new_step["not_answered_nodes"]:
                    state_in_new_step["not_answered_nodes"].remove(worker_action_content)
                if user_answer_state == Known:
                    state_in_new_step["known_nodes"].add(worker_action_content)
                elif user_answer_state == UnKnown:
                    state_in_new_step["unknown_nodes"].add(worker_action_content)
                state_in_new_step["last_turn_q_node"] = q_node
                state_in_new_step["last_turn_a_node"] = a_node

                # generate agent action mask
                state_in_new_step["rl_action_mask"][worker_action_idx] = 0
                if state_in_new_step["rl_action_mask"].sum() == 0 or \
                        state_in_new_step["total_qa_turn"] >= MinFlattenRLQATurn:
                    state_in_new_step["rl_action_mask"][-2:] = 1

                if mode == "RuleWarmUp":
                    state_in_new_step["warm_up_action_mask"][worker_action_idx] = 0
                    if state_in_new_step["total_qa_turn"] >= MaxFlattenRuleQATurn or \
                            state_in_new_step["warm_up_action_mask"].sum() == 0:
                        state_in_new_step["warm_up_action_mask"][:-2] = 0
                        if len(state_in_new_step["known_nodes"]) >= len(state_in_new_step["unknown_nodes"]):
                            state_in_new_step["warm_up_action_mask"][NON_FRAUD] = 1
                        else:
                            state_in_new_step["warm_up_action_mask"][FRAUD] = 1

            self.state_tracker_field.append(state_in_new_step)
            self.generate_recent_dialogue_feature()


class StateTrackerHRL(StateTracker):
    def __init__(self, state_tracker_field):
        super(StateTrackerHRL, self).__init__(state_tracker_field)

    def generate_system_action(self,
                               manager_sample_idx,
                               manager_sample_content,
                               workers_sample_idx,
                               workers_sample_content,
                               user_identity_state,
                               user_sub_identity_state_dict):
        policy_mask = self.state_tracker_field[-1]["policy_mask"]
        policy_idx = policy_mask.argmax()

        if policy_idx == 0:
            system_action = {"manager": {"sample_idx": manager_sample_idx,
                                         "sample_content": manager_sample_content}}

            # record manager's decision and its state
            if manager_sample_content in [FRAUD, NON_FRAUD]:
                self.state_tracker_field[-1]["manager_decision"] = manager_sample_content
                if manager_sample_content == user_identity_state:
                    self.state_tracker_field[-1]["manager_success_state"] = True
                else:
                    self.state_tracker_field[-1]["manager_success_state"] = False
        else:
            worker_idx = policy_idx - 1
            system_action = {"worker": {"sample_idx": workers_sample_idx[worker_idx],
                                        "sample_content": workers_sample_content[worker_idx]}}

            # record worker's decision and its state
            if workers_sample_content[worker_idx] in [FRAUD, NON_FRAUD]:
                self.state_tracker_field[-1]["workers_decision"][worker_idx] = workers_sample_content[worker_idx]
                sub_identity_state = user_sub_identity_state_dict[
                    str(self.state_tracker_field[-1]["personal_nodes"][worker_idx])]
                if workers_sample_content[worker_idx] == sub_identity_state:
                    self.state_tracker_field[-1]["workers_success_state"][worker_idx] = True
                else:
                    self.state_tracker_field[-1]["workers_success_state"][worker_idx] = False

        self.state_tracker_field[-1]["system_action"] = system_action
        return system_action

    def generate_workers_reward(self):
        # call this function after the worker choose a terminal action
        policy_mask = self.state_tracker_field[-1]["policy_mask"]
        worker_idx = policy_mask.argmax() - 1

        worker_decision = self.state_tracker_field[-1]["workers_decision"][worker_idx]
        worker_success_state = self.state_tracker_field[-1]["workers_success_state"][worker_idx]

        if worker_success_state is True:
            if worker_decision == FRAUD:
                self.state_tracker_field[-1]["workers_reward"] += (WorkerRecognitionCorrect + WorkerBonus)
            else:
                self.state_tracker_field[-1]["workers_reward"] += WorkerRecognitionCorrect
        else:
            if worker_decision != FRAUD:
                self.state_tracker_field[-1]["workers_reward"] += (WorkerRecognitionWrong - WorkerBonus)
            else:
                self.state_tracker_field[-1]["workers_reward"] += WorkerRecognitionWrong

        # rollback to give turn punishment to each worker action
        for i in range(-2, -(len(self.state_tracker_field) + 1), -1):
            if self.state_tracker_field[i]["policy_mask"].argmax() == 0:
                break
            self.state_tracker_field[i]["workers_reward"] += ExploringPunish

    def _worker_detect_a_fraud(self):
        flag = False
        for worker_decision, worker_success_state in zip(self.state_tracker_field[-1]["workers_decision"],
                                                         self.state_tracker_field[-1]["workers_success_state"]):
            if worker_decision == FRAUD and worker_success_state is True:
                flag = True
                break
        return flag

    def _worker_detect_all_non_fraud(self):
        flag = True
        for worker_decision, worker_success_state in zip(self.state_tracker_field[-1]["workers_decision"],
                                                         self.state_tracker_field[-1]["workers_success_state"]):
            if not ((worker_decision == PLACE_HOLDER and worker_success_state == PLACE_HOLDER) or (
                    worker_decision == NON_FRAUD and worker_success_state is True)):
                flag = False
                break
        return flag

    def generate_manager_reward(self):
        # call this function after the manager execute an action
        manager_decision = self.state_tracker_field[-1]["manager_decision"]
        manager_success_state = self.state_tracker_field[-1]["manager_success_state"]

        if manager_decision in [FRAUD, NON_FRAUD]:
            if manager_success_state is True:
                if not (self._worker_detect_a_fraud() or self._worker_detect_all_non_fraud()):
                    self.state_tracker_field[-1]["manager_reward"] += (ManagerRecognitionCorrect + ManagerBonus)
                else:
                    self.state_tracker_field[-1]["manager_reward"] += ManagerRecognitionCorrect
            else:
                if self._worker_detect_a_fraud() or self._worker_detect_all_non_fraud():
                    self.state_tracker_field[-1]["manager_reward"] += (ManagerRecognitionWrong - ManagerBonus)
                else:
                    self.state_tracker_field[-1]["manager_reward"] += ManagerRecognitionWrong

        # rollback to give turn punishment to the last manager action
        worker_exploring_time = self.state_tracker_field[-1]["current_worker_exploring_turn"]

        if len(self.state_tracker_field) > 1:
            for i in range(-2, -(len(self.state_tracker_field) + 1), -1):
                if self.state_tracker_field[i]["policy_mask"].argmax() == 0:
                    self.state_tracker_field[i]["manager_reward"] += worker_exploring_time * ExploringPunish
                    break

    def move_a_step(self,
                    language_generator,
                    user,
                    dialogue_recorder,
                    manager_sample_idx,
                    manager_sample_content,
                    manager_action_prob,
                    workers_sample_idx,
                    workers_sample_content,
                    workers_action_prob,
                    mode):
        """
        In current state S_{t}, the system execute the system action.
        And we get the new dialogue state S_{t+1}.
        Then, we generate all mask for the new state S_{t+1} based on dialogue context.
        :param language_generator: LanguageGenerator Class
        :param user: UserSimulator Class
        :param dialogue_recorder: DialogueRecorder Class or None
        :param manager_sample_idx:
        :param manager_sample_content:
        :param manager_action_prob:
        :param workers_sample_idx:
        :param workers_sample_content:
        :param workers_action_prob:
        :param mode: indicate rule based warm up or RL
        :return:
        """
        # just for debug
        self.state_tracker_field[-1]["manager_action_prob"] = manager_action_prob
        self.state_tracker_field[-1]["workers_action_prob"] = workers_action_prob

        # pad to the longest episode
        self.state_tracker_field[-1]["manager_sample_idx"] = manager_sample_idx
        self.state_tracker_field[-1]["workers_sample_idx"] = workers_sample_idx

        if self.state_tracker_field[-1]["episode_not_end"] is False:
            # pad to the longest episode
            self.state_tracker_field.append(copy.deepcopy(self.state_tracker_field[-1]))
        else:
            # generate the system action in the current valid dialogue state first
            system_action = self.generate_system_action(manager_sample_idx, manager_sample_content,
                                                        workers_sample_idx, workers_sample_content,
                                                        user.user_identity_state, user.user_sub_identity_state_dict)
            # and then inherit information of it
            state_in_new_step = copy.deepcopy(self.state_tracker_field[-1])

            state_in_new_step["total_exploring_turn"] += 1
            if state_in_new_step["total_exploring_turn"] >= MaxExploringTimeStep:
                # terminal hrl recursively in the next step by force
                state_in_new_step["rl_manager_action_mask"][:-2] = 0
                state_in_new_step["rl_manager_action_mask"][-2:] = 1
                state_in_new_step["rl_workers_action_mask"][:, :-2] = 0
                state_in_new_step["rl_workers_action_mask"][:, -2:] = 1

            # generate question
            q_node, a_node = language_generator.generate_question(system_action)

            # user give answer
            user_answer, user_answer_state = user.answer(q_node, a_node)

            # record dialogue
            if dialogue_recorder is not None:
                if "manager" in system_action.keys():
                    if system_action["manager"]["sample_content"] not in [FRAUD, NON_FRAUD]:
                        dialogue_recorder.record_manager_action_prob(manager_action_prob)
                    else:
                        dialogue_recorder.record_manager_action_prob(manager_action_prob,
                                                                     decision=system_action["manager"]["sample_content"])
                if "manager" in system_action.keys():
                    dialogue_recorder.record_key_actions(manager_action=system_action["manager"]["sample_content"],
                                                         worker_decision=None)
                elif "worker" in system_action.keys() and \
                        system_action["worker"]["sample_content"] in [FRAUD, NON_FRAUD]:
                    dialogue_recorder.record_key_actions(manager_action=None,
                                                         worker_decision=system_action["worker"]["sample_content"])

                if user_answer is not None:
                    dialogue_recorder.record_turn(language_generator, q_node=q_node, a_node=a_node,
                                                  user_answer=user_answer)
                elif "manager" in system_action.keys() and system_action["manager"]["sample_content"] in [FRAUD,
                                                                                                          NON_FRAUD]:
                    dialogue_recorder.record_turn(language_generator,
                                                  decision=system_action["manager"]["sample_content"])

            if user_answer is not None:
                state_in_new_step["total_qa_turn"] += 1

            # some information needs to be filled
            # we delete the useless time step in rollout
            state_in_new_step["dialogue_feature"] = PLACE_HOLDER
            state_in_new_step["manager_sample_idx"] = PLACE_HOLDER
            state_in_new_step["workers_sample_idx"] = PLACE_HOLDER
            state_in_new_step["system_action"] = PLACE_HOLDER
            state_in_new_step["reward_mask"] = PLACE_HOLDER
            state_in_new_step["manager_reward"] = np.zeros((1,), dtype=np.float32)
            state_in_new_step["workers_reward"] = np.zeros((len(state_in_new_step["personal_nodes"]),),
                                                           dtype=np.float32)

            # manager and work are the HRL agents
            # the manager_action_content and worker_action_content correspond to node id in graph or terminal action id
            # the manager_action_idx and worker_action_idx correspond to action idx in logits
            manager_action = system_action.get("manager", {"sample_idx": None, "sample_content": None})
            worker_action = system_action.get("worker", {"sample_idx": None, "sample_content": None})

            manager_action_idx = manager_action["sample_idx"]
            manager_action_content = manager_action["sample_content"]
            worker_action_idx = worker_action["sample_idx"]
            worker_action_content = worker_action["sample_content"]

            if manager_action_content in [FRAUD, NON_FRAUD]:
                """  terminal actions (Manager)  """
                # terminal reward for manager
                self.generate_manager_reward()

                # end this episode
                state_in_new_step["episode_not_end"] = False
            elif worker_action_idx is None:
                """  high level explore (Manager)  """
                state_in_new_step["valid_workers_num"] += 1

                # turn punishment for manager
                self.generate_manager_reward()

                # reset the turn counter for worker
                state_in_new_step["current_worker_exploring_turn"] = 0

                # generate for dialogue feature
                state_in_new_step["explored_nodes"].add(manager_action_content)
                state_in_new_step["last_turn_q_node"] = None
                state_in_new_step["last_turn_a_node"] = None
                if manager_action_content in state_in_new_step["not_explored_nodes"]:
                    state_in_new_step["not_explored_nodes"].remove(manager_action_content)

                # get new manager action mask
                state_in_new_step["rl_manager_action_mask"][manager_action_idx] = 0
                if mode == "RuleWarmUp":
                    state_in_new_step["warm_up_manager_action_mask"][manager_action_idx] = 0

                # get new policy mask
                policy_mask = [0] + [0 for _ in state_in_new_step["personal_nodes"]]
                worker_idx = state_in_new_step["personal_nodes"].index(manager_action_content)
                policy_mask[worker_idx + 1] = 1
                state_in_new_step["policy_mask"] = np.asarray(policy_mask, dtype=np.int32)
            elif user_answer is None:
                """  move from low level policy to high level policy (Worker)  """
                # generate workers rewards in current sub episode
                self.generate_workers_reward()

                # generate manager action mask
                if worker_action_content == FRAUD or state_in_new_step["rl_manager_action_mask"].sum() == 0:
                    state_in_new_step["rl_manager_action_mask"][-2:] = 1

                if mode == "RuleWarmUp":
                    if worker_action_content == FRAUD:
                        state_in_new_step["warm_up_manager_action_mask"][:-2] = 0
                        state_in_new_step["warm_up_manager_action_mask"][FRAUD] = 1
                    elif state_in_new_step["warm_up_manager_action_mask"].sum() == 0:
                        state_in_new_step["warm_up_manager_action_mask"][NON_FRAUD] = 1

                # generate for dialogue feature
                state_in_new_step["last_turn_q_node"] = None
                state_in_new_step["last_turn_a_node"] = None

                # get new policy mask
                policy_mask = [1] + [0 for _ in state_in_new_step["personal_nodes"]]
                state_in_new_step["policy_mask"] = np.asarray(policy_mask, dtype=np.int32)
            else:
                """  low level explore (Worker)  """
                state_in_new_step["current_worker_exploring_turn"] += 1

                worker_idx = state_in_new_step["policy_mask"].argmax() - 1
                state_in_new_step["workers_qa_turn"][worker_idx] += 1

                # generate for dialogue feature
                state_in_new_step["explored_nodes"].add(worker_action_content)
                if worker_action_content in state_in_new_step["not_explored_nodes"]:
                    state_in_new_step["not_explored_nodes"].remove(worker_action_content)
                if worker_action_content in state_in_new_step["not_answered_nodes"]:
                    state_in_new_step["not_answered_nodes"].remove(worker_action_content)
                if user_answer_state == Known:
                    state_in_new_step["known_nodes"].add(worker_action_content)
                    state_in_new_step["workers_counter"][worker_idx]["Known"] += 1
                elif user_answer_state == UnKnown:
                    state_in_new_step["unknown_nodes"].add(worker_action_content)
                    state_in_new_step["workers_counter"][worker_idx]["UnKnown"] += 1
                state_in_new_step["last_turn_q_node"] = q_node
                state_in_new_step["last_turn_a_node"] = a_node

                # generate workers action mask
                state_in_new_step["rl_workers_action_mask"][worker_idx, worker_action_idx] = 0
                if state_in_new_step["rl_workers_action_mask"][worker_idx].sum() == 0 or \
                        state_in_new_step["workers_qa_turn"][worker_idx] >= MinWorkerQATurn:
                    state_in_new_step["rl_workers_action_mask"][worker_idx, -2:] = 1
                if state_in_new_step["current_worker_exploring_turn"] >= MaxWorkerExploringTimeStep:
                    # terminal current worker in the next step by force
                    state_in_new_step["rl_workers_action_mask"][worker_idx, :-2] = 0

                if mode == "RuleWarmUp":
                    state_in_new_step["warm_up_workers_action_mask"][worker_idx, worker_action_idx] = 0
                    if state_in_new_step["current_worker_exploring_turn"] >= MaxWorkerExploringTimeStep or \
                            state_in_new_step["warm_up_workers_action_mask"][worker_idx].sum() == 0 or \
                            abs(state_in_new_step["workers_counter"][worker_idx]["UnKnown"] -
                                state_in_new_step["workers_counter"][worker_idx]["Known"]) >= MinDifference:
                        state_in_new_step["warm_up_workers_action_mask"][worker_idx, :-2] = 0
                        if state_in_new_step["workers_counter"][worker_idx]["UnKnown"] <= \
                                state_in_new_step["workers_counter"][worker_idx]["Known"]:
                            state_in_new_step["warm_up_workers_action_mask"][worker_idx, NON_FRAUD] = 1
                        else:
                            state_in_new_step["warm_up_workers_action_mask"][worker_idx, FRAUD] = 1

            state_in_new_step["reward_mask"] = np.copy(state_in_new_step["policy_mask"])
            if state_in_new_step["episode_not_end"] is False:
                state_in_new_step["reward_mask"] = state_in_new_step["reward_mask"] * 0

            self.state_tracker_field.append(state_in_new_step)
            self.generate_recent_dialogue_feature()
