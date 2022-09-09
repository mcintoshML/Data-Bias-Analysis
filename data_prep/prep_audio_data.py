import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel,delayed
import librosa
import librosa.display

from engine import utils
from constants import JUST_DIR,ICBHI_DIR,BASE_DATA_DIR
from constants import MEL_SPEC_SZ

LUNG_SOUND_DIR = '/cluster/projects/mcintoshgroup/BiasAnalysis/AudioBias/LungSounds/'

def get_csv(which='icbhi'):
    if which == 'just':
        df = pd.DataFrame()
        df['Path'] = glob('%s*.wav'%(JUST_DIR))
        df['name'] = [x.split('/')[-1].split('.')[0] for x in df.Path]
        df['PatientID'] = [x.split('_')[0] for x in df.name] 
        df['Label'] = [x.split(',')[0].split('_')[1].lower() for x in df.name] 
        mapping = {'n':'Normal','asthma':'Abnormal','copd':'Abnormal','pneumonia':'Abnormal'}
        df.Label = df.Label.replace(mapping)
    elif which == 'icbhi':
        df = pd.DataFrame()
        df['Path'] = glob('%s*.wav'%(ICBHI_DIR))
        df['name'] = [x.split('/')[-1].split('.')[0] for x in df.Path]
        df['PatientID'] = [x.split('_')[0] for x in df.name] 
        df.PatientID = df.PatientID.astype(np.int)
        df1 = pd.read_csv('%sICBHI_Labels.txt'%(ICBHI_DIR),header=None,names=['PatientID','Label'],delimiter='\t')
        df = pd.merge(df,df1,on='PatientID',how='left')
        df.Label = df.Label.str.lower()
        mapping = {'healthy':'Normal','asthma':'Abnormal','copd':'Abnormal','pneumonia':'Abnormal'}
        df.Label = df.Label.replace(mapping)
    df = df[df.Label.isin(['Normal','Abnormal'])]
    df['Normal'] = df.Label.str.contains('Normal')
    df['Abnormal'] = df.Label.str.contains('Abnormal')
    return df

def crop_border(img,border_color=255,sz=None):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,border_color,255,cv2.THRESH_BINARY_INV)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = img[y:y+h,x:x+w]
    if sz:
        crop = cv2.resize(crop,(sz,sz))
    return crop

def save_mel_spectrogram_image(path,num_sec=5,img_sz=224,target_folder=None):
    name = path.split('/')[-1].replace('.wav','')
    save_path = '%s%s.png'%(target_folder,name)

    y,sr = librosa.load(path)
    y = librosa.util.fix_length(y,size=sr*num_sec)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr,fmax=8000, ax=ax)
    plt.axis('off')
    fig.savefig(save_path,dpi=300,bbox_inches='tight')
    plt.close(fig)    

    img = cv2.imread(save_path)
    img = crop_border(img,border_color=250,sz=img_sz)
    cv2.imwrite(save_path,img)

def preprocess_and_save(which='icbhi',num_sec=5):
    df = get_csv(which=which)
    target_folder = utils.get_prepared_path('%sAudioLungSounds/mel_spec/%s/'%(BASE_DATA_DIR,which))
    Parallel(n_jobs=4,prefer='threads')(delayed(save_mel_spectrogram_image)(path=path,num_sec=num_sec,target_folder=target_folder,img_sz=MEL_SPEC_SZ) for path in tqdm(df.Path,desc='%s_%s'%(which,'melspec')))

def save_audio_csv(which):
    df = get_csv(which=which)
    path = utils.get_prepared_path('%sAudioBias/LungSounds/%s.csv.gz'%(BASE_DATA_DIR,which))
    df.to_csv(path,compression='gzip',index=None)
    print(df)


def main():
    save_audio_csv(which='icbhi')
    save_audio_csv(which='just')
    preprocess_and_save(which='icbhi',num_sec=5)
    preprocess_and_save(which='just',num_sec=5)