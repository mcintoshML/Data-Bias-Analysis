import pandas as pd
import numpy as np
import cv2
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader,Dataset
from joblib import Parallel,delayed
from tqdm import tqdm

from engine import utils
from constants import LUNG_DIS,PROCESSED_MEL_DIR
from constants import MEAN_STD
from dataloaders import data_tfms 


class Audio2Image(Dataset):
    def __init__(self,which='just_train',shuffle_pixels=False,num_samples=None,preload_to_ram=False,
        dtype=torch.float32,disable_tqdm=False,tfms=None,return_path=False,filter_unilabels=False,normalize=True,**kwargs):

        self.which = which
        self.shuffle_pixels = shuffle_pixels
        self.num_samples = num_samples
        self.normalize = normalize
        self.preload_to_ram = preload_to_ram
        self.disable_tqdm = disable_tqdm

        self.paths,self.labels = self.get_paths_labels()

        if self.normalize:
            mean,std = MEAN_STD[which.replace('_valid','_train')]
            self.normalize_fn = transforms.Normalize(mean,std)
        
        self.tensorize = data_tfms.ToTensor(dtype=dtype)
        self.randomize = data_tfms.Randomize2(p=1,dims=(0,1,2))

        # Smaller dataset - Faster to load to ram and process
        if self.preload_to_ram:
            self.X = self.preload_data()

    def get_paths_labels(self):
        path = 'data/csvs/%s.csv.gz'%self.which
        df = pd.read_csv(path)
        if self.num_samples:
            n = min(len(df),self.num_samples)
            df = df.sample(n=n,random_state=42)

        paths = df.Path.values
        ds_name = 'icbhi' if 'icbhi' in self.which else 'just'
        names = np.array([x.split('/')[-1].split('.')[0] for x in df.Path.values])
        paths = ['%s%s/%s.png'%(PROCESSED_MEL_DIR,ds_name,x) for x in names]        
        labels = df[LUNG_DIS].values
        return paths,labels

    def __len__(self):
        return len(self.labels)

    def preload_data(self):
        res = Parallel(n_jobs=-1,prefer='threads')(delayed(self.load_item)(path) for path in tqdm(self.paths,leave=False,desc='Preloading_%s'%self.which,disable=self.disable_tqdm)) 
        res = np.concatenate([np.expand_dims(x, 0) for x in res])
        return res        

    def load_item(self,path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # img = np.moveaxis(img, -1, 0)
        return img

    def __getitem__(self,idx):
        path,label = self.paths[idx],self.labels[idx]
        if self.preload_to_ram:
            img = self.X[idx]

        else:
            img = self.load_item(path)
        
        img = self.tensorize(img)
        img = img.permute((2, 0, 1))

        if self.normalize:
            img = self.normalize_fn(img)
        
        if self.shuffle_pixels:
            img = self.randomize(img)
        return img,label


def main():

    # calculate_mean_std = True
    calculate_mean_std = False
    # ds_name = 'icbhi_train'
    ds_name = 'just_train'

    
    if calculate_mean_std:
        # Caclulate mean and std using this and save to constants for new datasets
        ds = Audio2Image(which=ds_name,shuffle_pixels=False,normalize=False,num_samples=None,preload_to_ram=True)
        dl = DataLoader(ds,batch_size=100,num_workers=32,shuffle=False,pin_memory=False)
        utils.calculate_running_mean_std(dl,verbose=1)
    else:
        # visualize dataloaders
        with utils.set_seed(42):
            for shuffle_pixels in (False,True):
                ds = Audio2Image(which=ds_name,shuffle_pixels=shuffle_pixels,normalize=True,num_samples=100,preload_to_ram=True)
                dl = DataLoader(ds,batch_size=16,num_workers=8,shuffle=True,pin_memory=False)
                for idx,(x,y) in enumerate(dl):
                    print(ds.which,ds.shuffle_pixels,idx,x.shape,y.shape,utils.show_tensor_stats(x))
                    path = utils.get_prepared_path('visualize_dl/%s_shuffle_%s/%s.jpg'%(ds.which,ds.shuffle_pixels,str(idx).zfill(5)))
                    save_image(x,path,normalize=True,scale_each=False)

                    if idx == 2:
                        break