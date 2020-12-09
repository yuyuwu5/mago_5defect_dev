import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "..", "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")


TRAIN_DIR = os.path.join(DATA_DIR, "Train")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")

DEV_DIR = os.path.join(DATA_DIR, "Dev")
DEV_CSV = os.path.join(DATA_DIR, "dev.csv")


TRAIN_DEFECT_DIR = os.path.join(DATA_DIR, "defect_train")
DEV_DEFECT_DIR = os.path.join(DATA_DIR, "defect_dev")

TRAIN_FEATURE_PATH = "train_features.pt"
EVAL_FEATURE_PATH = "eval_features.pt"


DEFECT_TYPE = ['不良-乳汁吸附', '不良-機械傷害','不良-炭疽病', '不良-著色不佳', '不良-黑斑病']
EVAL_DIR = os.path.join(DATA_DIR, "eval")

