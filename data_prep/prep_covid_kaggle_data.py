import os
import cv2
from glob import glob,iglob
from tqdm import tqdm
from joblib import Parallel,delayed

from engine import utils
from constants import COVID_KAGGLE_DIR,COVID_XRAY_SZ
from constants import BASE_DATA_DIR


def resize_covid_kaggle_images():
    
    dst_folder = utils.get_prepared_path('%scovid_kaggle_256/%s/'%(BASE_DATA_DIR,COVID_KAGGLE_DIR.split('/')[-2]))

    def resize_save(src):
        dst = src.replace(COVID_KAGGLE_DIR,dst_folder)
        if os.path.exists(dst):
            return True
        utils.create_folder(dst)
        try:
            img = cv2.imread(src,0)
            img = cv2.resize(img,(COVID_XRAY_SZ,COVID_XRAY_SZ))
            status = cv2.imwrite(dst,img)
            return status
        except Exception as e:
            print(src,e)
            return False

    paths = iglob('%s*/*.png'%COVID_KAGGLE_DIR)
    utils.create_folder('%scovid_kaggle_256/'%(COVID_KAGGLE_DIR))
    results = Parallel(n_jobs=-1,prefer='threads')(delayed(resize_save)(path) for path in tqdm(paths,desc='Preparing Covid Kaggle Data'))
    utils.unique(results)

def main():
    resize_covid_kaggle_images()