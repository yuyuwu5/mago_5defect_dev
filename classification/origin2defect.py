import cv2
import numpy as np
import pandas as pd

from config import *

def originImg2Defect(mode="train"):
    if mode == "train":
        target_dir = TRAIN_DIR
        target_csv = TRAIN_CSV
        target_out = TRAIN_DEFECT_DIR
    elif mode == "dev":
        target_dir = DEV_DIR
        target_csv = DEV_CSV
        target_out = DEV_DEFECT_DIR
    else:
        print("Wrong Argument")
        exit(0)

    label = pd.read_csv(target_csv, header=None)
    all_type = []

    for defect_name in DEFECT_TYPE:
        defect_dir = os.path.join(target_out, defect_name)
        print(defect_dir)
        if not os.path.exists(defect_dir):
            print("mkdir ", defect_dir)
            os.mkdir(defect_dir)

    for idx in range(label.shape[0]):
        data = label.iloc[idx].dropna().tolist()
        img_name = data[0]
        img = cv2.imread(os.path.join(target_dir, img_name))
        for j in range(1, len(data), 5):
            x, y, dx, dy, disease = data[j:j+5]
            x, y, dx,dy = int(x), int(y), int(dx), int(dy)
            cropped_img = img[y:y+dy, x:x+dx]
            cv2.imwrite(os.path.join(target_out, disease, img_name[:-4]+"_"+str(j)+".jpg"), cropped_img)
        

    return

originImg2Defect(mode="dev")
