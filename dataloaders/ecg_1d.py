import pandas as pd
import numpy as np
import cv2
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader,Dataset
from joblib import Parallel,delayed
from tqdm import tqdm
import matplotlib.pyplot as plt

from engine import utils
from constants import PTBXL_DIS,PTBXL_DIR
from constants import LUDB_DIS_CL,LUDB_DIR
from constants import MEAN_STD,BASE_DATA_DIR
from dataloaders import data_tfms 


class ECG_Dataset(Dataset):
    def __init__(self,which='ptbxl_train',folds=[9,10],shuffle_pixels=False,num_samples=None,preload_to_ram=False,
        dtype=torch.float32,disable_tqdm=False,tfms=None,return_path=False,filter_unilabels=False,normalize=True,
        add_height_axis=False,
        **kwargs):

        self.which = which
        self.folds = folds
        self.shuffle_pixels = shuffle_pixels
        self.num_samples = num_samples
        self.normalize = normalize
        self.preload_to_ram = preload_to_ram
        self.disable_tqdm = disable_tqdm
        
        # Helps convert it to 2D array of 1 x 1000 with 12 channels 
        # For compatibility with torchvision normalization code
        self.add_height_axis = add_height_axis

        self.paths,self.labels = self.get_paths_labels()

        if self.normalize:
            mean,std = MEAN_STD[which.replace('_valid','_train')]
            self.normalize_fn = transforms.Normalize(mean,std)
        
        self.tensorize = data_tfms.ToTensor(dtype=dtype)
        self.randomize = data_tfms.Randomize2(p=1,dims=range(12))

        # Smaller dataset - Faster to load to ram and process
        if self.preload_to_ram:
            self.X = self.preload_data()

    def get_paths_labels(self):
        if 'ptbxl' in self.which:
            path = 'data/csvs/ptbxl.csv.gz'
            df = pd.read_csv(path)
            if '_train' in self.which:
                df = df[~df.strat_fold.isin(self.folds)]
            else:
                df = df[df.strat_fold.isin(self.folds)]
        elif 'ludb' in self.which:
            #Using the whole thing since its for validation only
            path = 'data/csvs/ludb.csv.gz'
            df = pd.read_csv(path)

        if self.num_samples:
            n = min(len(df),self.num_samples)
            df = df.sample(n=n,random_state=42)

        paths = df.Path.values

        if 'ptbxl' in self.which:
            paths = np.array(['%srecord100_npz/%s.npz'%(BASE_DATA_DIR,x.replace(PTBXL_DIR,'')) for x in paths])
        elif 'ludb' in self.which:
            paths = np.array(['%sdata_npz/%s.npz'%(BASE_DATA_DIR,x.replace(LUDB_DIR,'')) for x in paths])

        labels = df[PTBXL_DIS].values
        return paths,labels

    def __len__(self):
        return len(self.labels)

    def preload_data(self):
        res = Parallel(n_jobs=-1,prefer='threads')(delayed(self.load_item)(path) for path in tqdm(self.paths,leave=False,desc='Preloading_%s'%self.which,disable=self.disable_tqdm)) 
        res = np.concatenate([np.expand_dims(x, 0) for x in res])
        return res        

    def load_item(self,path):
        x = np.load(path)['signal']
        x = np.moveaxis(x,0,1)
        x = np.expand_dims(x,1) # C x H x W = 12 x 1 x 1000
        return x

    def __getitem__(self,idx):
        path,label = self.paths[idx],self.labels[idx]
        if self.preload_to_ram:
            img = self.X[idx]

        else:
            img = self.load_item(path)
        
        img = self.tensorize(img)

        if self.normalize:
            img = self.normalize_fn(img)
        
        if self.shuffle_pixels:
            img = self.randomize(img)

        if not self.add_height_axis:
            img = img.squeeze()

        return img,label


def main():

    # calculate_mean_std = True
    calculate_mean_std = False
    # ds_name = 'ptbxl_train'
    ds_name = 'ludb_train'

    
    if calculate_mean_std:
        # Caclulate mean and std using this and save to constants for new datasets
        # ds = ECG_Dataset(which=ds_name,folds=[9,10],shuffle_pixels=False,normalize=False,num_samples=None,preload_to_ram=True,add_height_axis=True)
        ds = ECG_Dataset(which=ds_name,folds=None,shuffle_pixels=False,normalize=False,num_samples=None,preload_to_ram=True,add_height_axis=True)
        dl = DataLoader(ds,batch_size=16,num_workers=32,shuffle=False,pin_memory=False)
        utils.calculate_running_mean_std(dl,verbose=1)
    else:
        # visualize dataloaders

        for shuffle_pixels in (False,True):
            with utils.set_seed(42):
                # ds = ECG_Dataset(which=ds_name,folds=[9,10],shuffle_pixels=shuffle_pixels,normalize=True,num_samples=100,preload_to_ram=True)
                ds = ECG_Dataset(which=ds_name,folds=None,shuffle_pixels=shuffle_pixels,normalize=True,num_samples=100,preload_to_ram=True)
                dl = DataLoader(ds,batch_size=16,num_workers=8,shuffle=True,pin_memory=False)
                for idx,(x,y) in enumerate(dl):
                    print(ds.which,ds.shuffle_pixels,idx,x.shape,y.shape,torch.mean(x,dim=(1,2)),torch.std(x,dim=(1,2)))

                    # # Save one sample
                    x = x[0].numpy().T
                    df = pd.DataFrame(x)
                    df.plot(subplots=True,legend=None)
                    path = utils.get_prepared_path('visualize_dl/%s_shuffle_%s/%s.jpg'%(ds.which,ds.shuffle_pixels,str(idx).zfill(5)))
                    plt.savefig(path,dpi=150,bbox_inches='tight')
                    plt.close()

                    if idx == 2:
                        break