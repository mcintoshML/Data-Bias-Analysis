import os
import pydicom
import numpy as np
import pandas as pd
from glob import glob,iglob
from tqdm import tqdm
from itertools import chain
import skimage.transform as sktf
from joblib import Parallel,delayed

from engine import utils
from constants import BASE_DATA_DIR,COVID_XRAY_SZ
from constants import COVID_INT_DIR1,COVID_INT_DIR2


def resize_covid_int_dcms():

    dst_folder = utils.get_prepared_path('%scovid_int_256/'%(BASE_DATA_DIR))

    def resize_save(src):
        if COVID_INT_DIR1 in src:
            dst = src.replace(COVID_INT_DIR1,'%scovid_int_256/'%(BASE_DATA_DIR)).replace('.dcm','.npz')
        elif COVID_INT_DIR2 in src:
            dst = src.replace(COVID_INT_DIR2,'%scovid_int_256/'%(BASE_DATA_DIR)).replace('.dcm','.npz')
        else:
            print(src)
            return False

        if os.path.exists(dst):
            return True
        try:
            img = pydicom.dcmread(src).pixel_array
            img = sktf.resize(img,[COVID_XRAY_SZ,COVID_XRAY_SZ],preserve_range=True,mode='constant',cval=0.0,anti_aliasing=True)
            img = img.astype(np.float32)
            utils.create_folder(dst)
            np.savez_compressed(dst,img=img)
            return True
        except Exception as e:
            print(src,e)
            return False

    df = pd.read_csv('data/csvs/covid_internal_train.csv.gz')
    paths = df.Path.values
    results = Parallel(n_jobs=-1)(delayed(resize_save)(path) for path in tqdm(paths,desc='Preparing CoVID Internal Data'))
    utils.unique(results)

def main():
    resize_covid_int_dcms()