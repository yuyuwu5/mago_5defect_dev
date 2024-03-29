import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "..", "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")


TRAIN_DIR = os.path.join(DATA_DIR, "Train")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_CROP_CSV = os.path.join(DATA_DIR, "train_crop.csv")

DEV_DIR = os.path.join(DATA_DIR, "Dev")
DEV_CSV = os.path.join(DATA_DIR, "dev.csv")
DEV_CROP_CSV = os.path.join(DATA_DIR, "eval_crop.csv")


TEST_DIR = os.path.join(DATA_DIR, "Test")
TEST_ORDER = os.path.join(DATA_DIR, 'Test_UploadSheet.csv')
TEST_BOX = os.path.join(DATA_DIR, 'Test_mangoXYWH.csv')


DEFECT_TYPE = [
        '不良-乳汁吸附', 
        '不良-機械傷害', 
        '不良-炭疽病',
        '不良-著色不佳', 
        '不良-黑斑病', 
        ]

DEFECT_SUBMIT_ID = {
        "D1": DEFECT_TYPE[0],
        "D2": DEFECT_TYPE[1],
        "D3": DEFECT_TYPE[2],
        "D4": DEFECT_TYPE[3],
        "D5": DEFECT_TYPE[4],
        }
EVAL_DIR = os.path.join(DATA_DIR, "eval")

