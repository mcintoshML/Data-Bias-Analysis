import pandas as pd
import numpy as np
import h5py
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader,Dataset

from engine import utils
from constants import ILD_DIS 
from constants import MEAN_STD
from dataloaders import data_tfms 


class RadiologyDataset3D(Dataset):
    def __init__(self,which='ild_TR_train',shuffle_pixels=False,num_samples=None,preload_to_ram=False,
        dtype=torch.float32,disable_tqdm=False,tfms=None,return_path=False,filter_unilabels=False,normalize=True,**kwargs):

        self.which = which
        self.shuffle_pixels = shuffle_pixels
        self.num_samples = num_samples
        self.normalize = normalize

        self.paths,self.labels = self.get_paths_labels()

        if self.normalize:
            mean,std = MEAN_STD[which.replace('_valid','_train')]
            self.normalize_fn = transforms.Normalize(mean,std)
        
        self.tensorize = data_tfms.ToTensor(dtype=dtype)
        self.randomize = data_tfms.Randomize(p=1,dims=(0,))

    def get_paths_labels(self):
        path = 'data/csvs/%s.csv.gz'%self.which
        df = pd.read_csv(path)
        if self.num_samples:
            n = min(len(df),self.num_samples)
            df = df.sample(n=n,random_state=42)


        paths = df.Path.values
        labels = df[ILD_DIS].values
        return paths,labels

    def __len__(self):
        return len(self.labels)

    def load_item(self,path):
        with h5py.File(path, "r") as f:
            img = np.array(f['img']).astype(np.float32)
        return img

    def __getitem__(self,idx):
        path,label = self.paths[idx],self.labels[idx]
        img = self.load_item(path)
        img = self.tensorize(img)
        if self.normalize:
            img = self.normalize_fn(img)
        if self.shuffle_pixels:
            img = self.randomize(img)
        return img,label


def main():

    calculate_mean_std = False
    # ds_name = 'ild_plan_train'
    ds_name = 'ild_TR_train'

    
    if calculate_mean_std:
        # Caclulate mean and std using this and save to constants for new datasets
        ds = RadiologyDataset3D(which=ds_name,shuffle_pixels=False,normalize=False,num_samples=None)
        dl = DataLoader(ds,batch_size=500,num_workers=32,shuffle=False,pin_memory=False)
        utils.calculate_running_mean_std(dl,verbose=1)
    else:
        # visualize dataloaders
        with utils.set_seed(42):
            for shuffle_pixels in (False,True):
                ds = RadiologyDataset3D(which=ds_name,shuffle_pixels=shuffle_pixels,normalize=True,num_samples=100)
                dl = DataLoader(ds,batch_size=16,num_workers=8,shuffle=True,pin_memory=False)
                for idx,(x,y) in enumerate(dl):
                    print(idx,x.shape,y.shape,utils.show_tensor_stats(x))
                    x = x[0,0,:,:,:]
                    x = torch.moveaxis(x, -1, 0)
                    x = x.unsqueeze(1)
                    path = utils.get_prepared_path('visualize_dl/%s_shuffle_%s/%s.jpg'%(ds.which,ds.shuffle_pixels,str(idx).zfill(5)))
                    save_image(x,path,normalize=True,scale_each=False)

                    if idx == 2:
                        break