import h5py
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from engine import utils
from constants import ILD_DIR,BASE_DATA_DIR


COMMON_ID = 'SeriesInstanceUID'

def get_paths(which='train'):
    if which == 'train':
        paths = glob('%sILD_Training_HR/*.h5'%(ILD_DIR))    
    elif which == 'plan':
        paths = glob('%sILD_Planning_HR_2/*.h5'%(ILD_DIR))    
    df = pd.DataFrame({'Path':paths})
    df[COMMON_ID] = [x.split('/')[-1].replace('_CT.h5','') for x in df.Path]
    return df

def get_csv(which):
    if which == 'dicom_info':
        path = '%sdcm_df_3mm_Train_anonID.csv'%(ILD_DIR)
        df = pd.read_csv(path)
        df = df.drop_duplicates(subset=COMMON_ID, keep="first")
        return df

def get_h5_item(path,col_name='classes'):
    with h5py.File(path, "r") as f:
        value = np.array(f[col_name])[0][0]
    return value

def prep_data(which='train'):
    if which == 'train':
        df1 = get_paths(which=which)
        df2 = get_csv(which='dicom_info')
        df = pd.merge(df1,df2,on=[COMMON_ID],how='left')
        df = df[['Path',COMMON_ID,'PatientID']]
    elif which == 'plan':
        df = get_paths(which=which)
        # Each entry comes from unique patient in Plan
        df['PatientID'] = ['PID_Plan_%d'%idx for idx in range(len(df))]
        df = df[['Path',COMMON_ID,'PatientID']]

    df['Label'] = Parallel(n_jobs=-1)(delayed(get_h5_item)(path,col_name='classes') for path in tqdm(df.Path))
    
    # Labels were initially flipped
    # df['Normal'] = (df.Label==0)*1
    # df['Abnormal'] = (df.Label==1)*1

    df['Normal'] = (df.Label==1)*1
    df['Abnormal'] = (df.Label==0)*1

    print(df.Label.value_counts())
    print(df.PatientID.nunique())

    path = utils.get_prepared_path('%sILD/%s.csv.gz'%(BASE_DATA_DIR,which))
    df.to_csv(path,compression='gzip',index=None)
    print(df.Label.value_counts())
    print(df)

def main():
    prep_data(which='train')
    prep_data(which='plan')