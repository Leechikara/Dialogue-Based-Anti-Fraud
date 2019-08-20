# coding = utf-8
import torch
import os

DATA_ROOT = os.path.abspath("../data")

# for data set split
train_size = 706
test_size = 100
dev_size = 100

# for pad
# NodePad depends on the valid num of nodes
RelationPad = 0
EdgePad = (0, 0, 0)

PLACE_HOLDER = None

# for manager and worker terminal actions, to keep in accordance to other actions
# we insert [FRAUD, NON_FRAUD] in the tail of action space
FRAUD = -2
NON_FRAUD = -1

# simulate user know or do not know about a triple
Known = 1
UnKnown = -1
NotClear = 0
ShowUnknown = "Unknown"
UnKnownUtterance = "我不清楚"

NegativeSampledAnswerNum = 2
Options = ["A", "B", "C", "D"]

# for user type sample
User_Type_Weights = {"Type-4 Fraud": 1,
                     "Type-3 Fraud": 1,
                     "Type-2 Fraud": 1,
                     "Type-1 Fraud": 1,
                     "Non-Fraud": 4}
Personal_Information_Fraud_Weights = {"company": 2, "university": 2, "live_in": 1, "born_in": 1}

EPS = 1e-6

# Turn exploring time step constrain
# Note: the exploring time steps are not the same as dialogue turns,
# punish exploring time steps will punish dialogue turns, not vice versa
MaxExploringTimeStep = 40
MaxWorkerExploringTimeStep = 10
MinFlattenRLQATurn = 8
MaxFlattenRuleQATurn = 10

# For rule based warm ups
MinDifference = 3
MinWorkerQATurn = 3

# Rewards
WorkerBonus = 0
ManagerBonus = 0
# give bonus in hrl model setting, if not, the rl will collapse, just a trick!
# WorkerBonus = 2
# ManagerBonus = 1.5
ManagerRecognitionCorrect = 3
ManagerRecognitionWrong = -3
WorkerRecognitionCorrect = 1
WorkerRecognitionWrong = -1
ExploringPunish = -0.1
WorkerRewardDiscount = 0.99
ManagerRewardDiscount = 0.999

# config for torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for node feature
PERSONAL_NODE_FILED = 4
ONE_STEP_NODE_FILED = 1
SE_FREQS_FILED = 10
DEGREE_FILED = 10
STATIC_FEATURE_SIZE = PERSONAL_NODE_FILED + ONE_STEP_NODE_FILED + SE_FREQS_FILED + DEGREE_FILED
DYNAMIC_FEATURE_SIZE = 7
Init_Node_Feature_Size = STATIC_FEATURE_SIZE + DYNAMIC_FEATURE_SIZE
ManagerStateRest = 4 + 2 * 4 + (MaxWorkerExploringTimeStep + 1) * 3 * 4 + (MaxExploringTimeStep + 1)
WorkersStateRest = (MaxWorkerExploringTimeStep + 1) * 3 + (MaxWorkerExploringTimeStep + 1) + (
        MaxWorkerExploringTimeStep + 1)

# for flatten rl, this node idx is the padded query node
Pad_Query_Node = 1000
