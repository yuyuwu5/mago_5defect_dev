import os
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "..", "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model_fat")


ALL_DIR = os.path.join(DATA_DIR, 'All')
ALL_CSV = os.path.join(DATA_DIR, 'all.csv')
ALL_CROP_CSV = os.path.join(DATA_DIR, 'all_crop.csv')
ALL_ONE_HOT = os.path.join(DATA_DIR, 'all-one-hot.npy')


COUNT_TOTAL_IMG = len(os.listdir(ALL_DIR))-1
total_img_id_list = list(range(COUNT_TOTAL_IMG))
random.Random(502087).shuffle(total_img_id_list)

FOLD_NUM = 5
count_img_in_one_fold = COUNT_TOTAL_IMG // FOLD_NUM

FOLD = [ total_img_id_list[0 : count_img_in_one_fold],
        total_img_id_list[1*count_img_in_one_fold : 2*count_img_in_one_fold],
        total_img_id_list[2*count_img_in_one_fold : 3*count_img_in_one_fold],
        total_img_id_list[3*count_img_in_one_fold : 4*count_img_in_one_fold],
        total_img_id_list[4*count_img_in_one_fold : ]
        ]

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

DEFECT_PATCH = [
        os.path.join(DATA_DIR, 'defect_train', DEFECT_TYPE[0]),
        os.path.join(DATA_DIR, 'defect_train', DEFECT_TYPE[1]),
        os.path.join(DATA_DIR, 'defect_train', DEFECT_TYPE[2]),
        os.path.join(DATA_DIR, 'defect_train', DEFECT_TYPE[3]),
        os.path.join(DATA_DIR, 'defect_train', DEFECT_TYPE[4]),
        ]
