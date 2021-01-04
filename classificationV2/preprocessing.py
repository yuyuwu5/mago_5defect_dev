import pandas as pd
import numpy as np
from config import *

def preprocessing(CROP_CSV, mode='train'):
    train_csv = pd.read_csv(CROP_CSV)
    train_img_names = train_csv.iloc[:,0].tolist()
    train_bbox = list(zip(
        train_csv.iloc[:,1].tolist(),
        train_csv.iloc[:,2].tolist(),
        train_csv.iloc[:,3].tolist(),
        train_csv.iloc[:,4].tolist(),
        ))
    train_one_hot = [[0,0,0,0,0] for i in range(len(train_img_names))]
    for i in range(len(train_img_names)):
        tmp_data = train_csv.iloc[i, 5:].dropna().tolist()
        l_tmp = len(tmp_data)
        for j in range(0, l_tmp, 5):
            disease = tmp_data[j+4]
            train_one_hot[i][DEFECT_TYPE.index(disease)] = 1
    train_pos_count = np.array(train_one_hot).sum(axis=0)
    multilabel = pd.concat([
        pd.DataFrame(data=np.array(train_img_names), columns=['image_names']),
        train_csv.iloc[:,1:5], 
        pd.DataFrame(data=np.array(train_one_hot), columns=DEFECT_TYPE)
    ], axis=1)
    #multilabel.columns[0] = 'img_names'
    #print(multilabel)
    multilabel.to_csv(os.path.join(DATA_DIR, mode+'label.csv'), index=False)

if __name__ == '__main__':
    preprocessing(TRAIN_CROP_CSV, mode='train')
    preprocessing(DEV_CROP_CSV, mode='dev')