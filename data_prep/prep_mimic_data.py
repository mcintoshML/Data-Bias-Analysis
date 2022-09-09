import os
import pydicom
import numpy as np
from glob import glob,iglob
from tqdm import tqdm
import skimage.transform as sktf
from joblib import Parallel,delayed

from engine import utils
from constants import BASE_DATA_DIR,CXRAY_SZ,MIMIC_DIR


def resize_mimic_dcms():

    dst_folder = utils.get_prepared_path('%sdcm320/'%(BASE_DATA_DIR))

    def resize_save(src):
        dst = src.replace(MIMIC_DIR,dst_folder).replace('.dcm','.npz')
        if os.path.exists(dst):
            return True
        utils.create_folder(dst)
        try:
            img = pydicom.dcmread(src).pixel_array
            img = sktf.resize(img,[CXRAY_SZ,CXRAY_SZ],preserve_range=True,mode='constant',cval=0.0,anti_aliasing=True)
            img = img.astype(np.float32)
            utils.create_folder(dst)
            np.savez_compressed(dst,img=img)
            return True
        except Exception as e:
            print(src,e)
            return False

    paths = iglob('%s*/*/*/*.dcm'%(MIMIC_DIR))
    results = Parallel(n_jobs=-1,prefer='threads')(delayed(resize_save)(path) for path in tqdm(paths,desc='Preparing MIMIC Data'))
    utils.unique(results)

def main():
    resize_mimic_dcms()