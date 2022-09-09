import cv2
import numpy as np
import pandas as pd
from joblib import delayed,Parallel
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader,Dataset

from engine import utils
from constants import BASE_DATA_DIR
from dataloaders import data_tfms 
from constants import CXRAY_DIS,DIS_CLASSES,MEAN_STD
from constants import MIMIC_DIR,NIH_DIR,CHEXPERT_DIR
from constants import COVID_INT_DIR1,COVID_INT_DIR2


def convert_to_preprocessed_paths(path):
    if 'CheXpert-v1.0-small' in path:
        path = '%schexpert_320/%s'%(BASE_DATA_DIR,path)
        return path,'chexpert_train'
    elif 'mimic' in path:
        path = path.replace(MIMIC_DIR,'').replace('.dcm','.npz')
        path = '%sdcm320/%s'%(BASE_DATA_DIR,path)
        return path,'mimic_train'
    elif 'nih' in path:
        name = path.split('/')[-1].replace('.png','.jpg')
        path = '%snih_images_320/%s'%(NIH_DIR,name) 
        return path,'nih_train'
    elif 'COVID-19_Radiography_Dataset' in path:
        path = '%scovid_kaggle_256/%s'%(BASE_DATA_DIR,path)
        return path,'covid_kaggle_train'
    elif ('COVID-imaging' in path or 'COVID-PositiveOnly' in path):
        path = path.replace(COVID_INT_DIR1,'%scovid_int_256/'%BASE_DATA_DIR)
        path = path.replace(COVID_INT_DIR2,'%scovid_int_256/'%BASE_DATA_DIR)
        path = path.replace('.dcm','.npz')
        return path,'covid_internal_train'    

class RadiologyDataset2D(Dataset):
    def __init__(self,which='chexpert_train',shuffle_pixels=False,num_samples=None,preload_to_ram=False,
        dtype=torch.float32,disable_tqdm=False,tfms=None,return_path=False,filter_unilabels=False,normalize=False,img_sz=None,**kwargs):

        self.which = which
        self.shuffle_pixels = shuffle_pixels
        self.num_samples = num_samples
        self.normalize = normalize
        self.img_sz = img_sz
        self.dis_cl = DIS_CLASSES[which.replace('_valid','').replace('_train','')]
        self.paths,self.labels,self.ds_names = self.get_paths_labels()
        
        self.tensorize = data_tfms.ToTensor(dtype=dtype)
        self.randomize = data_tfms.Randomize(p=1,dims=(0,)) 

        if self.normalize:
            self.normalize_fns = {
                ds_name:transforms.Normalize(mean,std) for ds_name,(mean,std) in MEAN_STD.items()
            }

    def get_paths_labels(self):
        path = 'data/csvs/%s.csv.gz'%self.which
        df = pd.read_csv(path)
        if self.num_samples:
            n = min(len(df),self.num_samples)
            df = df.sample(n=n,random_state=42)
        paths = df.Path.values
        res = np.array(Parallel(n_jobs=-1,prefer='threads')(delayed(convert_to_preprocessed_paths)(path) for path in paths))
        paths,ds_names = np.hsplit(res,2)
        paths,ds_names = paths.flatten(),ds_names.flatten()
        labels = df[self.dis_cl].values
        return paths,labels,ds_names

    def __len__(self):
        return len(self.labels)

    def load_item(self,path):
        if '.npz' in path[-4:]:
            img = np.load(path,allow_pickle=True)['img']
        else:
            img = cv2.imread(path,0)
        img = img[None,:,:]
        return img

    def __getitem__(self,idx):
        path,label = self.paths[idx],self.labels[idx]
        
        try:
            img = self.load_item(path)
        except Exception as e:
            # print(path)
            return self.__getitem__(idx-1)

        img = self.tensorize(img)

        if self.normalize:
            # Normalize each dataset sample in combined datasets
            # This ensures that data has mean=0, std=1 even when dicoms and images are in same batch            
            ds_name = self.ds_names[idx]
            normalize_fn = self.normalize_fns[ds_name]
            img = normalize_fn(img)

        if self.shuffle_pixels:
            img = self.randomize(img)

        return img,label

def main():

    calculate_mean_std = True
    # calculate_mean_std = False

    ds_name = 'chexpert_80_20_train'
    # ds_name = 'mimic_train'
    # ds_name = 'nih_train'
    # ds_name = 'combined_train' #Chexpert + MIMIC
    # ds_name = 'combined_nihmimic_train'
    # ds_name = 'covid_kaggle_train'
    # ds_name = 'covid_internal_train'

    
    if calculate_mean_std:
        # Caclulate mean and std using this and save to constants for new datasets
        ds = RadiologyDataset2D(which=ds_name,shuffle_pixels=False,normalize=False,num_samples=None)
        dl = DataLoader(ds,batch_size=500,num_workers=32,shuffle=False,pin_memory=False)
        utils.calculate_running_mean_std(dl,verbose=1)
    else:
        # visualize dataloaders
        with utils.set_seed(42):
            for shuffle_pixels in (False,True):
                ds = RadiologyDataset2D(which=ds_name,shuffle_pixels=shuffle_pixels,normalize=True,num_samples=100)
                dl = DataLoader(ds,batch_size=16,num_workers=8,shuffle=True,pin_memory=False)

                for idx,(x,y) in enumerate(dl):
                    print(idx,x.shape,y.shape,utils.show_tensor_stats(x))

                    path = utils.get_prepared_path('visualize_dl/%s_shuffle_%s/%s.jpg'%(ds.which,ds.shuffle_pixels,str(idx).zfill(5)))
                    save_image(x,path,normalize=True,scale_each=True)

                    if idx == 2:
                        break