# coding = utf-8

import numpy as np
import os
import json
from src.config import DATA_ROOT
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def get_performance(model_setting):
    success_rate_list = list()
    turn_list = list()

    with open(os.path.join(DATA_ROOT, "checkpoints", model_setting, "running_record.json"), "r") as f:
        running_record = json.load(f)

    for mode in ["RuleWarmUp", "RL"]:
        if mode in running_record.keys():
            for record in running_record[mode]["running_record"].values():
                success_rate_list.append(record["success"])
                turn_list.append(record["turn"])

    return success_rate_list, turn_list


def mean_std(y, windows=10):
    mean = list()
    std = list()
    for i in range(0, len(y)):
        mean.append(np.mean(np.asarray(y[i - windows + 1 if i - windows + 1 >= 0 else 0:i + 1])))
        std.append(np.std(np.asarray(y[i - windows + 1 if i - windows + 1 >= 0 else 0:i + 1])))
    return np.asarray(mean), np.asarray(std)


def learning_curve():
    model_settings = ["ghrl", "hrl", "grl"]
    success_rate = dict()
    average_turns = dict()

    for model_setting in model_settings:
        success_rate_list, turn_list = get_performance(model_setting)

        success_rate[model_setting] = {}
        original_success_rate = np.asarray(success_rate_list, dtype=np.float)
        mean_success_rate, std_success_rate = mean_std(original_success_rate)
        success_rate[model_setting]["original"] = original_success_rate
        success_rate[model_setting]["mean"] = mean_success_rate
        success_rate[model_setting]["std"] = std_success_rate

        average_turns[model_setting] = {}
        original_turn = np.asarray(turn_list, dtype=np.float)
        mean_turn, std_turn = mean_std(original_turn)
        average_turns[model_setting]["original"] = original_turn
        average_turns[model_setting]["mean"] = mean_turn
        average_turns[model_setting]["std"] = std_turn

    assert success_rate["ghrl"]["original"].shape[0] == success_rate["grl"]["original"].shape[0] == \
           success_rate["hrl"]["original"].shape[0]
    epochs = success_rate["ghrl"]["original"].shape[0]
    x = np.arange(1, epochs + 1, step=1, dtype=np.int)
    labels = ["Full-S", "HP-S", "MP-S"]
    colors = ["b", "g", "r"]

    # draw success rate curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(xlim=[x[0], x[-1]], ylim=[0.4, 0.9],xlabel="Simulation Epoch", ylabel="Accuracy")
    for model_setting, label, color in zip(model_settings, labels, colors):
        ax.plot(x, success_rate[model_setting]["mean"], label=label, color=color, lw=1)
        ax.fill_between(x,
                        (success_rate[model_setting]["mean"] - success_rate[model_setting]["std"]),
                        (success_rate[model_setting]["mean"] + success_rate[model_setting]["std"]),
                        color=color, alpha=0.15)
    ax.legend(loc="lower right")
    plt.xticks([20, 70, 120, 170, 220, 270, 320])
    plt.show()
    fig.savefig(os.path.join(DATA_ROOT, "Figure", "accuracy_curve.pdf"), dpi=fig.dpi)

    # draw turns curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(xlim=[x[0], x[-1]], xlabel="Simulation Epoch", ylabel="Average Turns")
    for model_setting, label, color in zip(model_settings, labels, colors):
        ax.plot(x, average_turns[model_setting]["mean"], label=label, color=color, lw=1)
        ax.fill_between(x,
                        average_turns[model_setting]["mean"] - average_turns[model_setting]["std"],
                        average_turns[model_setting]["mean"] + average_turns[model_setting]["std"],
                        color=color, alpha=0.1)
    ax.legend(loc="best")
    plt.xticks([20, 70, 120, 170, 220, 270, 320])
    plt.show()
    fig.savefig(os.path.join(DATA_ROOT, "Figure", "turns_curve.pdf"), dpi=fig.dpi)


if __name__ == "__main__":
    learning_curve()
