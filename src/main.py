# coding = utf-8

import sys, argparse, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('__file__'), os.path.pardir)))
from src.DialogueSystem.world import World


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_setting", default=None, type=str)
    parser.add_argument("--rand_seed", default=44)
    parser.add_argument("--record_dialogue", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_hierarchy_policy", action="store_true")
    parser.add_argument("--use_graph_based_state_tracker", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_rule_based_system", action="store_true")
    parser.add_argument("--trained_model", default=None, type=str)
    parser.add_argument("--rl", action="store_true")
    parser.add_argument("--warm_up", action="store_true")
    parser.add_argument("--eval_interval", default=1)
    parser.add_argument("--eval_epoch", default=10)
    parser.add_argument("--rl_epoch", default=300)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--rl_learning_rate", default=1e-4)
    parser.add_argument("--warm_up_learning_rate", default=3e-4)
    parser.add_argument("--warm_up_epoch", default=20)
    parser.add_argument("--warm_up_interval", default=5)
    parser.add_argument("--new_node_emb_size_list", nargs='*', type=int,
                        help="The node embed size of GNN in next few iterations.")
    parser.add_argument("--msg_agg", default="max")
    parser.add_argument("--global_agg_size", default=100)
    parser.add_argument("--score_method", default="concat")
    parser.add_argument("--max_clip", default=5)
    parser.add_argument("--entropy_coef", default=0.1)

    args = parser.parse_args()
    return args


def args_verify(args):
    if args.model_setting not in ["hrl", "grl", "ghrl"]:
        raise ValueError("Model setting can be only hrl, grl and ghrl.")
    if args.model_setting == "hrl" and (not args.use_hierarchy_policy or args.use_graph_based_state_tracker):
        raise ValueError("In hrl setting, we do not use graph based state tracker but use hierarchy policy.")
    if args.model_setting == "grl" and (args.use_hierarchy_policy or not args.use_graph_based_state_tracker):
        raise ValueError("In grl setting, we do not use hierarchy policy but use graph based state tracker.")
    if args.model_setting == "ghrl" and (not args.use_hierarchy_policy or not args.use_graph_based_state_tracker):
        raise ValueError("In ghrl setting, we use graph based state tracker and hierarchy policy.")
    if not args.use_graph_based_state_tracker and args.new_node_emb_size_list is not None:
        raise ValueError("If not using graph based state tracker, the new node embedding list should be None.")
    if args.use_graph_based_state_tracker and args.new_node_emb_size_list is None:
        raise ValueError("If using graph based state tracker, the new node embedding list should not be None.")
    if args.test_rule_based_system and not args.test:
        raise ValueError("Testing rule based system only exists in test mode.")
    if args.score_method not in ["dotted", "general", "concat"]:
        raise ValueError("Score(attention) methods should be dotted, general or concat.")
    if args.msg_agg not in ["sum", "avg", "max"]:
        raise ValueError("Message aggregation methods should be sum, avg or max.")


def main():
    args = parse()
    args_verify(args)
    world = World(args)
    world.run()


if __name__ == "__main__":
    main()
