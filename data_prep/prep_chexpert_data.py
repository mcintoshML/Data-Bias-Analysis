import os
import cv2
from glob import glob,iglob
from tqdm import tqdm
from joblib import Parallel,delayed

from engine import utils
from constants import BASE_DATA_DIR,CXRAY_SZ,CHEXPERT_DIR


def resize_cxp_images(split):

    dst_folder = utils.get_prepared_path('%schexpert_320/%s/'%(BASE_DATA_DIR,CHEXPERT_DIR.split('/')[-2]))

    def resize_save(src):
        dst = src.replace(CHEXPERT_DIR,dst_folder)
        if os.path.exists(dst):
            return True
        utils.create_folder(dst)
        try:
            img = cv2.imread(src,0)
            img = cv2.resize(img,(CXRAY_SZ,CXRAY_SZ))
            status = cv2.imwrite(dst,img)
            return status
        except Exception as e:
            print(src,e)
            return False

    paths = iglob('%s%s/*/*/*.jpg'%(CHEXPERT_DIR,split))
    results = Parallel(n_jobs=-1,prefer='threads')(delayed(resize_save)(path) for path in tqdm(paths,desc='Preparing Chexpert Data'))
    utils.unique(results)

def main():
    resize_cxp_images(split='valid')
    resize_cxp_images(split='train')