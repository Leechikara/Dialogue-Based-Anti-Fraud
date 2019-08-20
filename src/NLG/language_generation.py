# coding = utf-8
import os
import json
import re
import random
import copy
from src.config import DATA_ROOT, FRAUD, NON_FRAUD, NegativeSampledAnswerNum, Options, ShowUnknown, UnKnownUtterance

with open(os.path.join(DATA_ROOT, "idx2r.json"), "r") as f:
    idx2r = json.load(f)

with open(os.path.join(DATA_ROOT, "answersLibrary.json"), "r") as f:
    answers_library = json.load(f)

with open(os.path.join(DATA_ROOT, "languageTemplates.json"), "r") as f:
    templates = json.load(f)


class LanguageGenerator(object):
    def __init__(self, language_generation_filed):
        self.idx2r = idx2r
        self.language_generation_filed = language_generation_filed
        self.query_entity = None
        self.answers_library = answers_library
        self.templates = templates

    def generate_question(self, system_action):
        if "worker" in system_action.keys() and system_action["worker"]["sample_content"] not in [FRAUD, NON_FRAUD]:
            return self.query_entity["node_id"], system_action["worker"]["sample_content"]
        elif "manager" in system_action.keys() and system_action["manager"]["sample_content"] not in [FRAUD, NON_FRAUD]:
            query_entity_node_id = system_action["manager"]["sample_content"]
            node = self.language_generation_filed["idx2node"][str(query_entity_node_id)]
            self.query_entity = {"node": node, "node_id": query_entity_node_id}
            return None, None
        else:
            return None, None

    def _generate_sys_nl(self, h, r, t):
        """
        Assume the h r t have been transferred to NL
        return: natural language question, candidates, the correct answer option
        """
        question = self.templates[str(r)]
        question = re.sub(r"\$\S\$", h, question)

        # To avoid user using exclusive method,
        # the sampled negative answers should have similar appearance to the correct answer.
        all_candidates = copy.deepcopy(self.answers_library[r])
        all_candidates.remove(t)
        for c in all_candidates:
            if t.find(c) != -1 or c.find(t) != -1:
                all_candidates.remove(c)

        answers_candidates = random.sample(all_candidates, NegativeSampledAnswerNum)
        answers_candidates.append(t)
        random.shuffle(answers_candidates)
        answers_candidates.append("不太清楚")
        candidates = "  ".join([" ".join([option, answer]) for option, answer in zip(Options, answers_candidates)])

        return question, candidates

    def generate_sys_nl(self, h, t):
        r = self.language_generation_filed["h_t_to_r"][str(h) + " " + str(t)]
        h = self.language_generation_filed["idx2node"][str(h)]
        t = self.language_generation_filed["idx2node"][str(t)]
        r = self.idx2r[str(r)]

        question, candidates = self._generate_sys_nl(h, r, t)
        return "    ".join([question, candidates])

    @staticmethod
    def generate_user_nl(user_answer):
        if user_answer != ShowUnknown:
            return user_answer
        else:
            return UnKnownUtterance
