import os
import cv2
from glob import glob
from tqdm import tqdm
from joblib import Parallel,delayed

from engine import utils
from constants import NIH_DIR,CXRAY_SZ


def resize_nih_images():
    
    def resize_save(src):
        name = src.split('/')[-1].replace('.png','.jpg')
        dst = '%snih_images_320/%s'%(NIH_DIR,name) 
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

    paths = glob('%simages/*.png'%NIH_DIR)
    utils.create_folder('%snih_images_320/'%(NIH_DIR))
    results = Parallel(n_jobs=-1,prefer='threads')(delayed(resize_save)(path) for path in tqdm(paths,desc='Preparing NIH Data'))
    utils.unique(results)

def main():
    resize_nih_images()