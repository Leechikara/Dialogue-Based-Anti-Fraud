# coding = utf-8
from src.config import FRAUD


class DialogueRecorder(object):
    def __init__(self):
        self.dialogue = list()
        self.manager_action_prob = list()
        self.key_actions = list()

    def record_turn(self, language_generator, **kwargs):
        if "decision" not in kwargs.keys():
            sys_nl = language_generator.generate_sys_nl(kwargs["q_node"], kwargs["a_node"])
            user_nl = language_generator.generate_user_nl(kwargs["user_answer"])
            self.dialogue.append({"System: ": sys_nl, "User: ": user_nl})
        else:
            self.dialogue.append({"System's Decision is: ": "Fraud" if kwargs["decision"] == FRAUD else "Not_Fraud"})

    def record_manager_action_prob(self, manager_action_prob, **kwargs):
        self.manager_action_prob.append(list(manager_action_prob))
        if "decision" in kwargs.keys():
            for i in range(5 - len(self.manager_action_prob)):
                self.manager_action_prob.append([0, 0, 0, 0, 0, 0])

    def record_key_actions(self, manager_action, worker_decision):
        assert manager_action is None or worker_decision is None
        if manager_action is not None:
            self.key_actions.append(("manager", manager_action))
        if worker_decision is not None:
            self.key_actions.append(("worker_decision", worker_decision))
