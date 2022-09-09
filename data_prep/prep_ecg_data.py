import pandas as pd
from tqdm import tqdm
from joblib import delayed,Parallel
import wfdb
import numpy as np
from constants import PTBXL_DIR,BASE_DATA_DIR,PTBXL_DIS
from constants import LUNG_DIS,LUDB_DIR
from engine import utils
from scipy import signal


# /cluster/projects/mcintoshgroup/Results/PTB-XL/ptbxl_splits.csv
def read_csv(which):
    if which == 'ptbxl':
        df = pd.read_csv('%sptbxl_splits.csv'%PTBXL_DIR)
        df = df.rename(columns={'patient_id':'PatientID','filename_lr':'Path'})
    elif which == 'ludb':
        df = pd.read_csv('%sludb_df.csv'%LUDB_DIR)
        df['Path'] = ['data/%s'%(x) for x in df.ID]
        df['PatientID'] = df.header_file.copy()
    return df

def process_csv(which):
    df = read_csv(which=which)
    for cl in PTBXL_DIS:
        df[cl] = pd.Series([cl in x for x in df.diagnostic_superclass])*1
    print(df.diagnostic_superclass.value_counts())
    print(df[PTBXL_DIS].sum())
    df.to_csv('data/csvs/%s.csv.gz'%which,index=None,compression='gzip')

def convert_dat2npz(path,which):
    if which == 'ptbxl':
        path = '%s%s'%(PTBXL_DIR,path)
        data = wfdb.rdsamp(path)[0]
        outpath = utils.get_prepared_path('%srecord100_npz/%s.npz'%(BASE_DATA_DIR,path.replace(PTBXL_DIR,'')))
        np.savez_compressed(outpath,signal=data)
    elif which == 'ludb':
        path = '%s%s'%(LUDB_DIR,path)
        data = wfdb.rdsamp(path)[0]
        data = signal.resample(data,1000)
        outpath = utils.get_prepared_path('%sdata_npz/%s.npz'%(BASE_DATA_DIR,path.replace(LUDB_DIR,'')))
        np.savez_compressed(outpath,signal=data)

def process_data(which):
    df = read_csv(which=which)
    Parallel(n_jobs=-1,prefer='threads')(delayed(convert_dat2npz)(path,which) for path in tqdm(df.Path))

def main():
    # read_csv(which='ptbxl')
    # read_csv(which='ludb')
    
    process_csv(which='ludb')
    process_csv(which='ptbxl')
    
    process_data(which='ptbxl')
    process_data(which='ludb')