import pandas as pd
import numpy as np
import wfdb
import ast
import torch
import torch.nn as nn

import torch, queue
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
#from .core import *
import collections,sys,traceback,threading

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numbers as num
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score

from sklearn.metrics import roc_auc_score, roc_curve, auc,f1_score, confusion_matrix
from torchvision import transforms, utils
#from dataHelpers import ECGTabularSet,ToTensor,Randomize
#from dataloader import DataLoader
#from pySegHelpers import trainModel
from matplotlib import pyplot as plt

import argparse
from distutils.util import strtobool
import random as rand
from datetime import datetime
from csv import DictWriter


todays_date=datetime.today().strftime("%d/%m/%Y")
print(f'Date: {todays_date}')

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--randomize',type=lambda x:bool(strtobool(x)), default=False, help='Randomize model')
parser.add_argument('-n', '--normalization', type=str,  help='signal normalization method [population, all_leads, individual_leads, all_leads_min_max,individual_leads_min_max]')
parser.add_argument('-f', '--filter',type=lambda x:bool(strtobool(x)), default=False, help='Yes to filtering for only patients with CD/HYP label ')
parser.add_argument('-wd', '--weight_decay',type=lambda x:bool(strtobool(x)), default=False, help='Add weight decay and switch to optim=SGD')
parser.add_argument('-two', '--two_class',type=lambda x:bool(strtobool(x)), default=False, help='two classes in model prediction')


args = parser.parse_args()


NORMALIZATION=args.normalization
RANDOMIZE=args.randomize
FILTER_PTS =args.filter
WEIGHT_DECAY=args.weight_decay
TWO_CLASS=args.two_class

SAVE_PTH_NAME = 'eval2class_' + 'ptbxl' +'_'+ str(NORMALIZATION) +'_'+ 'rand' + str(RANDOMIZE)+ '2class' + str(TWO_CLASS) + 'filter' + str(FILTER_PTS) + 'May3'
 
print(f' This model is saved at {SAVE_PTH_NAME}, normalized: {NORMALIZATION}, randomized: {RANDOMIZE}, filtered {FILTER_PTS}, weightdecay&sgd {WEIGHT_DECAY}, two class {TWO_CLASS}')

EPOCH=10


class ECGTabularSet(Dataset):
    def __init__(self, data,labels,lead_dict, NORMALIZATION):
        """
        Args:
            mat_file: Path to the matlab file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.labels = labels
        self.indicies = None
        self.lead_dict = lead_dict
        self.normalization = NORMALIZATION
        
        
    def setIndicies(self,indicies):
        self.indicies = indicies                 
    def __len__(self):
        if(self.indicies is not None):
            length = len(self.indicies)
        else:
            length = self.data.shape[0]
        return length

    def __getitem__(self, idx):
        if(self.indicies is not None):
                idx = self.indicies[idx]
        lead_dict = self.lead_dict

        sample = {'image':self.data[idx,...].copy(),
                  'labels':self.labels[idx,...].copy(),
                  'dfs':0,
                  'scalars':None,
                  'censored':0,
                  'index':idx}

        if self.normalization == 'population':
        
            ### Normalize image using population ecg mean and stddev
            norm_I = (sample['image'][0]-lead_dict['lead_I_avg'])/lead_dict['lead_I_std']
            norm_II = (sample['image'][1]-lead_dict['lead_II_avg'])/lead_dict['lead_II_std']
            norm_III = (sample['image'][2]-lead_dict['lead_III_avg'])/lead_dict['lead_III_std']
            norm_avr =(sample['image'][3]-lead_dict['avr_avg'])/lead_dict['avr_std']
            norm_avl =(sample['image'][4]-lead_dict['avl_avg'])/lead_dict['avl_std']
            norm_avf =(sample['image'][5]-lead_dict['avf_avg'])/lead_dict['avf_std']

            norm_v1 = (sample['image'][6]-lead_dict['v1_avg'])/lead_dict['v1_std']
            norm_v2 = (sample['image'][7]-lead_dict['v2_avg'])/lead_dict['v2_std']
            norm_v3 = (sample['image'][8]-lead_dict['v3_avg'])/lead_dict['v3_std']
            norm_v4 = (sample['image'][9]-lead_dict['v4_avg'])/lead_dict['v4_std']
            norm_v5 = (sample['image'][10]-lead_dict['v5_avg'])/lead_dict['v5_std']
            norm_v6 = (sample['image'][11]-lead_dict['v6_avg'])/lead_dict['v6_std']
            norm_ecg= np.stack((norm_I,norm_II,norm_III, norm_avr, norm_avl, norm_avf,norm_v1,norm_v2,norm_v3,
                                   norm_v4, norm_v5,norm_v6),axis=1)
            norm_ecg = np.swapaxes(norm_ecg, 0,1)   
            #sample['image'] = norm_ecg
            #print('normalized by population leads ')

        elif self.normalization == 'individual_leads':
            
            ### Normalize image using individual lead ecg mean and stddev
            norm_I = (sample['image'][0]-sample['image'][0].mean())/(sample['image'][0].std()+1e-10)
            norm_II = (sample['image'][1]-sample['image'][1].mean())/(sample['image'][1].std()+1e-10)
            norm_III = (sample['image'][2]-sample['image'][2].mean())/(sample['image'][2].std()+1e-10)
            norm_avr =(sample['image'][3]-sample['image'][3].mean())/(sample['image'][3].std()+1e-10)
            norm_avl =(sample['image'][4]-sample['image'][4].mean())/(sample['image'][4].std()+1e-10)
            norm_avf =(sample['image'][5]-sample['image'][5].mean())/(sample['image'][5].std()+1e-10)

            norm_v1 = (sample['image'][6]-sample['image'][6].mean())/(sample['image'][6].std()+1e-10)
            norm_v2 = (sample['image'][7]-sample['image'][7].mean())/(sample['image'][7].std()+1e-10)
            norm_v3 = (sample['image'][8]-sample['image'][8].mean())/(sample['image'][8].std()+1e-10)
            norm_v4 = (sample['image'][9]-sample['image'][9].mean())/(sample['image'][9].std()+1e-10)
            norm_v5 = (sample['image'][10]-sample['image'][10].mean())/(sample['image'][10].std()+1e-10)
            norm_v6 = (sample['image'][11]-sample['image'][11].mean())/(sample['image'][11].std()+1e-10)
            norm_ecg= np.stack((norm_I,norm_II,norm_III, norm_avr, norm_avl, norm_avf,norm_v1,norm_v2,norm_v3,
                                   norm_v4, norm_v5,norm_v6),axis=1)
            norm_ecg = np.swapaxes(norm_ecg, 0,1)   
            #sample['image'] = norm_ecg
            #print('normalized by each individual patient lead ')
            
            #norm_ecg= np.random.rand(12,1000)
            #accuracy of this should be 50%

        elif self.normalization == 'all_leads':
        ### Normalize image using patient 12 lead ecg mean and stddev
            all_leads_mean, all_leads_std = sample['image'].mean(), sample['image'].std()
            norm_I = (sample['image'][0]-all_leads_mean)/all_leads_std
            norm_II = (sample['image'][1]-all_leads_mean)/all_leads_std
            norm_III = (sample['image'][2]-all_leads_mean)/all_leads_std
            norm_avr =(sample['image'][3]-all_leads_mean)/all_leads_std
            norm_avl =(sample['image'][4]-all_leads_mean)/all_leads_std
            norm_avf =(sample['image'][5]-all_leads_mean)/all_leads_std

            norm_v1 = (sample['image'][6]-all_leads_mean)/all_leads_std
            norm_v2 = (sample['image'][7]-all_leads_mean)/all_leads_std
            norm_v3 = (sample['image'][8]-all_leads_mean)/all_leads_std
            norm_v4 = (sample['image'][9]-all_leads_mean)/all_leads_std
            norm_v5 = (sample['image'][10]-all_leads_mean)/all_leads_std
            norm_v6 = (sample['image'][11]-all_leads_mean)/all_leads_std

            norm_ecg= np.stack((norm_I,norm_II,norm_III, norm_avr, norm_avl, norm_avf,norm_v1,norm_v2,norm_v3,
                                   norm_v4, norm_v5,norm_v6),axis=1)
            norm_ecg = np.swapaxes(norm_ecg, 0,1)   
            #sample['image'] = norm_ecg
            #print('normalized by patient lead mean and std')
        
        elif self.normalization == 'all_leads_min_max':

        ### Normalize image using patient 12 lead ecg mean and stddev
            all_leads_min, all_leads_max = sample['image'].min(), sample['image'].max()
            norm_I = (sample['image'][0]-all_leads_min)/(all_leads_max-all_leads_min)
            norm_II = (sample['image'][1]-all_leads_min)/(all_leads_max-all_leads_min)
            norm_III = (sample['image'][2]-all_leads_min)/(all_leads_max-all_leads_min)
            norm_avr =(sample['image'][3]-all_leads_min)/(all_leads_max-all_leads_min)
            norm_avl =(sample['image'][4]-all_leads_min)/(all_leads_max-all_leads_min)
            norm_avf =(sample['image'][5]-all_leads_min)/(all_leads_max-all_leads_min)

            norm_v1 = (sample['image'][6]-all_leads_min)/(all_leads_max-all_leads_min)
            norm_v2 = (sample['image'][7]-all_leads_min)/(all_leads_max-all_leads_min)
            norm_v3 = (sample['image'][8]-all_leads_min)/(all_leads_max-all_leads_min)
            norm_v4 = (sample['image'][9]-all_leads_min)/(all_leads_max-all_leads_min)
            norm_v5 = (sample['image'][10]-all_leads_min)/(all_leads_max-all_leads_min)
            norm_v6 = (sample['image'][11]-all_leads_min)/(all_leads_max-all_leads_min)

            norm_ecg= np.stack((norm_I,norm_II,norm_III, norm_avr, norm_avl, norm_avf,norm_v1,norm_v2,norm_v3,
                                   norm_v4, norm_v5,norm_v6),axis=1)
            norm_ecg = np.swapaxes(norm_ecg, 0,1)   
            #sample['image'] = norm_ecg
            #print('normalized by patient lead mean and std')
        
        
        elif self.normalization == 'individual_leads_min_max':
    
            ### Normalize image using individual lead ecg mean and stddev
            norm_I = (sample['image'][0]-sample['image'][0].min())/((sample['image'][0].max()-sample['image'][0].min())+1e-10)
            norm_II = (sample['image'][1]-sample['image'][1].min())/((sample['image'][1].max()-sample['image'][1].min())+1e-10)
            norm_III = (sample['image'][2]-sample['image'][2].min())/((sample['image'][2].max()-sample['image'][2].min())+1e-10)
            norm_avr =(sample['image'][3]-sample['image'][3].min())/((sample['image'][3].max()-sample['image'][3].min())+1e-10)
            norm_avl =(sample['image'][4]-sample['image'][4].min())/((sample['image'][4].max()-sample['image'][4].min())+1e-10)
            norm_avf =(sample['image'][5]-sample['image'][5].min())/((sample['image'][5].max()-sample['image'][5].min())+1e-10)

            norm_v1 = (sample['image'][6]-sample['image'][6].min())/((sample['image'][6].max()-sample['image'][6].min())+1e-10)
            norm_v2 = (sample['image'][7]-sample['image'][7].min())/((sample['image'][7].max()-sample['image'][7].min())+1e-10)
            norm_v3 = (sample['image'][8]-sample['image'][8].min())/((sample['image'][8].max()-sample['image'][8].min())+1e-10)
            norm_v4 = (sample['image'][9]-sample['image'][9].min())/((sample['image'][9].max()-sample['image'][9].min())+1e-10)
            norm_v5 = (sample['image'][10]-sample['image'][10].min())/((sample['image'][10].max()-sample['image'][10].min())+1e-10)
            norm_v6 = (sample['image'][11]-sample['image'][11].min())/((sample['image'][11].max()-sample['image'][11].min())+1e-10)
            norm_ecg= np.stack((norm_I,norm_II,norm_III, norm_avr, norm_avl, norm_avf,norm_v1,norm_v2,norm_v3,
                                   norm_v4, norm_v5,norm_v6),axis=1)
            norm_ecg = np.swapaxes(norm_ecg, 0,1)   

        else:
            #print('leads not normalized')
            None
        
        sample['image'] = norm_ecg


        if self.transform:
           sample = self.transform(sample)

        return sample
    
    def getCounts(self):
        ndI = self[0]['labels'].ndim
        sz = self[0]['labels'].shape
        numC = sz[ndI-1]
        counts = np.zeros(numC)
        for i in range(len(self)):
            for c in range(0,numC):
                lbl = self[i]['labels'][...,c]
                if(lbl.max()>0):
                    counts[c] += 1
        return counts


    def getmeanstd(self,sampleCount=None):
        lastD = 2
            
        if sampleCount is None:
            sampleCount = len(self)
            sampleSet = range(sampleCount)
        else:
            sampleSet = np.random.permutation(range(sampleCount))[0:sampleCount]
        
        dims = tuple(i for i in range(1,lastD+1))
        mnSet = Parallel(n_jobs=self.num_cores,prefer='threads')(
                delayed(np.nanmean)(self[i]['image'],dims,keepdims=True) for i in sampleSet)
        
        stdSet = Parallel(n_jobs=self.num_cores,prefer='threads')(
                delayed(np.nanstd)(self[i]['image'],dims,keepdims=True) for i in sampleSet)
        
        mn = np.mean(mnSet,0)
        std = np.mean(stdSet,0)
        return mn, std
    
    def getLabels(self):
        if(self.indicies is not None):
            return self.labels[self.indicies,...]
        else:
            return self.labels

    def setTransform(self, transform):
        self.transform = transform
        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, c3D=False,activeChannels = None,numC=1,numClasses=2,device='cpu'):
        self.c3D = c3D
        self.numC = numC
        self.numClasses = numClasses
        self.activeChannels = activeChannels
        self.device = device
    def __call__(self, sample):
        out = sample
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if(self.activeChannels is not None):
            out['image'] = out['image'][self.activeChannels,...]            
        
#        sz = image.shape
#        if(self.c3D):
#            if(self.numC>1):
#                image = np.moveaxis(image,[0,1,2,3],[1,2,3,0])
#            image = image.reshape((self.numC,sz[0],sz[1],sz[2]),order='F')
#        else:
#            if(self.numC>1):
#                image = np.moveaxis(image,[0,1,2],[1,2,0])
#            image = image.reshape((self.numC,sz[0],sz[1]),order='F')
#            
#            
#        if(labels.ndim>1):
#            if(self.c3D):
#                #labels = np.moveaxis(labels,[0,1,2,3],[1,2,3,0])
#                #labels = np.moveaxis(labels,[0,1,2,3],[1,2,3,0])
#                #labels = labels.reshape((self.numClasses-1,sz[0],sz[1],sz[2]))
#                #labels = labels.reshape((self.numClasses-1,sz[0],sz[1],sz[2]),order='F')
#                labels = labels.reshape((self.numClasses-1,sz[0],sz[1],sz[2]),order='F')
#            else:
#                labels = np.moveaxis(labels,[0,1,2],[1,2,0])
#                labels = labels.reshape((self.numClasses-1,sz[0],sz[1]),order='F')
                
        #.pin_memory() is slower for multiple workers
        for key in out.keys():
            if(out[key] is not None and (isinstance(out[key],np.ndarray) or isinstance(out[key],np.generic) or isinstance(out[key],num.Number))):
                #print(key)
                out[key] = torch.from_numpy(np.asarray(out[key])).to(self.device)
        
        #Clean up the output for stacking
        removeList = []
        for k in out:
            if(out[k] is None):
                removeList.append(k)
        for k in removeList:
            del out[k]
        
        return out

class Randomize(object):
    """Normalize the image """
    def __init__(self, dims,p=1.0):
        self.dims = dims
        self.p = p
    def __call__(self, sample):
        out = sample
        if np.random.rand(1)[0] <= self.p:
            image = out['image']
            shp = image.shape
            idx = np.random.permutation(np.arange(np.prod(shp[1:])))
            for i in list(self.dims):
                #idx = np.random.permutation(np.arange(np.prod(shp[1:])))
                image[i,...] = image[i,...].reshape(-1)[idx].reshape(shp[1:])
    
            out['image'] = image
        return out


string_classes = (str, bytes)

def to_gpu(x):
    return x

def T(a, half=False, cuda=False):
    """
    Convert numpy array into a pytorch tensor. 
    if Cuda is available and USE_GPU=True, store resulting tensor in GPU.
    """
    if not torch.is_tensor(a):
        #a = np.array(np.ascontiguousarray(a))
        if a.dtype in (np.int8, np.int16, np.int32, np.int64):
            a = torch.from_numpy(a) #torch.LongTensor(a.astype(np.int64))
        elif a.dtype in (np.float32, np.float64):
            if half:
                a = torch.cuda.HalfTensor(a)
            else:
                a = torch.from_numpy(a)  #torch.FloatTensor(a)
        else:
            raise NotImplementedError(a.dtype)
    if cuda:
        a = to_gpu(a)
    return a

def chunk_iter(iterable, chunk_size):
    '''A generator that yields chunks of iterable, chunk_size at a time. '''
    while True:
        chunk = []
        try:
            for _ in range(chunk_size): chunk.append(next(iterable))
            yield chunk
        except StopIteration:
            if chunk: yield chunk
            break
def proc(batch):
    yield get_tensor(batch, False,False)

def get_tensor(batch, pin, half=False):
    if isinstance(batch, (np.ndarray, np.generic)):
        batch = T(batch, half=half, cuda=False).contiguous()
        if pin: batch = batch.pin_memory()
        return to_gpu(batch)
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, collections.Mapping):
        return {k: get_tensor(sample, pin, half) for k, sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [get_tensor(sample, pin, half) for sample in batch]
    elif isinstance(batch,torch.Tensor):
        if pin: batch = batch.pin_memory()
        return batch
    elif batch is None:
        return None
    raise TypeError(f"batch must contain numbers, dicts or lists; found {type(batch)}")


class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, pad_idx=0,
                 num_workers=None, pin_memory=False, drop_last=False, pre_pad=True, half=False,
                 transpose=False, transpose_y=False):
        self.dataset,self.batch_size,self.num_workers = dataset,batch_size,num_workers
        self.pin_memory,self.drop_last,self.pre_pad = pin_memory,drop_last,pre_pad
        self.transpose,self.transpose_y,self.pad_idx,self.half = transpose,transpose_y,pad_idx,half

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with '
                                 'batch_size, shuffle, sampler, and drop_last')

        if sampler is not None and shuffle:
            raise ValueError('sampler is mutually exclusive with shuffle')

        if batch_sampler is None:
            if sampler is None:
                sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        if num_workers is None:
            self.num_workers = num_cpus()

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __len__(self): return len(self.batch_sampler)

    def jag_stack(self, b):
        if len(b[0].shape) not in (1,2): return np.stack(b)
        ml = max(len(o) for o in b)
        if min(len(o) for o in b)==ml: return np.stack(b)
        res = np.zeros((len(b), ml), dtype=b[0].dtype) + self.pad_idx
        for i,o in enumerate(b):
            if self.pre_pad: res[i, -len(o):] = o
            else:            res[i,  :len(o)] = o
        return res

    def np_collate(self, batch):
        b = batch[0]
        if isinstance(b, (np.ndarray, np.generic)): return self.jag_stack(batch)
        elif isinstance(b, (int, float)): return np.array(batch)
        elif isinstance(b, string_classes): return batch
        elif isinstance(b, collections.Mapping):
            return {key: self.np_collate([d[key] for d in batch]) for key in b}
        elif isinstance(b, collections.Sequence):
            return [self.np_collate(samples) for samples in zip(*batch)]
        elif isinstance(b,torch.Tensor):
            #return torch.stack(batch,dim=0)
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = b.storage()._new_shared(numel)
                out = b.new(storage)
            return torch.stack(batch, 0, out=out)
        elif b is None:
            return batch
        raise TypeError(("batch must contain numbers, dicts or lists; found {}".format(type(b))))

    
    def get_batch(self, indices):
        res = self.np_collate([self.dataset[i] for i in indices])
        if self.transpose:   res[0] = res[0].T
        if self.transpose_y: res[1] = res[1].T
        return res

    def __iter__(self):
        
        if self.num_workers==0:
            for batch in map(self.get_batch, iter(self.batch_sampler)):
                yield get_tensor(batch, self.pin_memory, self.half)
        else:
             # Parallel(n_jobs=self.num_workers,prefer='threads')(delayed(proc)(batch)
             #                                   for batch in map(self.get_batch, iter(self.batch_sampler)))            
            with ThreadPoolExecutor(max_workers=self.num_workers) as e:
            #with ProcessPoolExecutor(max_workers=self.num_workers) as e:
                for batch in e.map(self.get_batch, iter(self.batch_sampler)):
                    yield get_tensor(batch, self.pin_memory, self.half)
            # avoid py3.6 issue where queue is infinite and can result in memory exhaustion
                # for c in chunk_iter(iter(self.batch_sampler), self.num_workers*10):
                #     for batch in e.map(self.get_batch, c):
                #         yield get_tensor(batch, self.pin_memory, self.half)

print('Starting\n')

if TWO_CLASS==True:
    class simpleECG(nn.Module):
        def __init__(self):
            super(simpleECG, self).__init__()
            k = 64
            self.c1 = nn.Conv1d(12,k,7,padding=3)
            self.b1 = nn.BatchNorm1d(k)
            self.R = nn.ReLU()
            
            M = 6
            modules = []
            for i in range(0,M):
                modules.append(nn.Conv1d(k,k*2,3,padding=1))
                modules.append(nn.BatchNorm1d(k*2))
                modules.append(nn.ReLU())
                modules.append(nn.AvgPool1d(2,2))
                k=k*2
            self.LayerBlocks = nn.ModuleList(modules)
            #self.fc = nn.Linear(k,5)
            self.fc = nn.Linear(k,2)
            self.GP = nn.AdaptiveAvgPool1d((1))
        def forward(self,x):
            x = self.R(self.b1(self.c1(x)))
            for i,L in enumerate(self.LayerBlocks):
                x = L(x)
            x = self.GP(x)
            x = torch.flatten(x, 1)
            return self.fc(x)
else:
    class simpleECG(nn.Module):
        def __init__(self):
            super(simpleECG, self).__init__()
            k = 64
            self.c1 = nn.Conv1d(12,k,7,padding=3)
            self.b1 = nn.BatchNorm1d(k)
            self.R = nn.ReLU()
            
            M = 6
            modules = []
            for i in range(0,M):
                modules.append(nn.Conv1d(k,k*2,3,padding=1))
                modules.append(nn.BatchNorm1d(k*2))
                modules.append(nn.ReLU())
                modules.append(nn.AvgPool1d(2,2))
                k=k*2
            self.LayerBlocks = nn.ModuleList(modules)
            self.fc = nn.Linear(k,5)
            #self.fc = nn.Linear(k,2)
            self.GP = nn.AdaptiveAvgPool1d((1))
        def forward(self,x):
            x = self.R(self.b1(self.c1(x)))
            for i,L in enumerate(self.LayerBlocks):
                x = L(x)
            x = self.GP(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

        
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data
#path = r'D:/Studies/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
#path= r'/cluster/projects/mcintoshgroup/PBT-ECG/'
path = r'/cluster/home/cathy.ongly/physionet.org/files/ptb-xl/1.0.1/'
sampling_rate=100

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
#X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# look at value counts for diagnostic superclass
#def to_1D(series):
#    return pd.Series([x for _list in series for x in _list])
#to_1D(Y['diagnostic_superclass']).value_counts()

# SELECT FOR ONLY HYP AND CD ROWS
### ADDD filtering argument
if FILTER_PTS==True:
    selection = ['HYP', 'CD']
    mask = Y.diagnostic_superclass.apply(lambda x: any(item for item in selection if item in x))
    Y = Y[mask]
    print('only rows with HYP/CD patients selected')
else:
    None
    print('all patient rows selected')

print('Shape of Y', Y.shape)
# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)
print('Shape of X', X.shape)

#*************BUILD LEAD DICTIONARY FOR NORMALIZATION***************
def lead_avg_std_dict(data):
    lead_I, lead_II, lead_III= [],[],[]
    avr,avl,avf = [],[],[]
    v1,v2,v3,v4,v5,v6 = [],[],[],[],[],[]
    for x,y in enumerate(data):
        if x<1000: 
            #print(y.shape)
            lead_I.append(y[:,0])
            lead_II.append(y[:,1])
            lead_III.append(y[:,2])
            avr.append(y[:,3])
            avl.append(y[:,4])
            avf.append(y[:,5])
            v1.append(y[:,6])
            v2.append(y[:,7])
            v3.append(y[:,8])
            v4.append(y[:,9])
            v5.append(y[:,10])
            v6.append(y[:,11])

    lead_avgs_std = {'lead_I_avg': (np.array(lead_I)).mean(),
                     'lead_I_std': (np.array(lead_I)).std(),
                     'lead_II_avg': (np.array(lead_II)).mean(),
                     'lead_II_std': (np.array(lead_II)).std(),
                     'lead_III_avg': (np.array(lead_III)).mean(),
                     'lead_III_std': (np.array(lead_III)).std(),

                     'avr_avg': (np.array(avr)).mean(),
                     'avr_std':(np.array(avr)).std(),
                     'avl_avg': (np.array(avl)).mean(),
                     'avl_std':(np.array(avl)).std(),
                     'avf_avg': (np.array(avf)).mean(),
                     'avf_std':(np.array(avf)).std(),

                     'v1_avg': (np.array(v1)).mean(),
                     'v1_std':(np.array(v1)).std(),
                     'v2_avg': (np.array(v2)).mean(),
                     'v2_std':(np.array(v2)).std(),
                     'v3_avg': (np.array(v3)).mean(),
                     'v3_std':(np.array(v3)).std(),
                     'v4_avg': (np.array(v4)).mean(),
                     'v4_std':(np.array(v4)).std(),
                     'v5_avg': (np.array(v5)).mean(),
                     'v5_std':(np.array(v5)).std(),
                     'v6_avg': (np.array(v6)).mean(),
                     'v6_std':(np.array(v6)).std(),
                    }  
    return lead_avgs_std
ptbxl_leads =  lead_avg_std_dict(X)

# Split data into train and test
test_fold = 10

# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[(Y.strat_fold == test_fold)].diagnostic_superclass

X_train = np.swapaxes(X_train,1,2)
X_test = np.swapaxes(X_test,1,2)

if TWO_CLASS==True:
    def encodeLabels(Y):
        #L = np.zeros((len(Y),5))
        L = np.zeros((len(Y),2))
        #C = ['CD','MI','HYP','STTC','NORM']
        C = ['CD','HYP']
        for idx,item in enumerate(Y):
            for ic,cd in enumerate(C):
                if cd in item:
                    L[idx,ic] = 1
        return L
    print('y_encode 2 classes')
else:
    def encodeLabels(Y):
        L = np.zeros((len(Y),5))
        #L = np.zeros((len(Y),2))
        C = ['CD','HYP', 'MI','STTC','NORM']
        #C = ['CD','HYP']
        for idx,item in enumerate(Y):
            for ic,cd in enumerate(C):
                if cd in item:
                    L[idx,ic] = 1
        return L
    print('y_encode 5 classes')

y_train_enc = encodeLabels(y_train)
y_test_enc = encodeLabels(y_test)

## Deep model bias

#Build dataloaders


trainSet = ECGTabularSet(X_train.astype(np.float32), y_train_enc.astype(np.float32),ptbxl_leads,NORMALIZATION=NORMALIZATION)
testSet = ECGTabularSet(X_test.astype(np.float32), y_test_enc.astype(np.float32),ptbxl_leads,NORMALIZATION=NORMALIZATION)
mbSize = 16
numWorkers = 0
LR = 1e-4
Count = 10
CudaON = True
device = 'cuda'

#WITH RANDOMIZATION 
if RANDOMIZE==True:
    transformSet = transforms.Compose([Randomize(range(0,12)),ToTensor(device=device)])
    print('data randomized')
#WITHOUT RANDOMIZATION
else:
    transformSet = transforms.Compose([ToTensor(device=device)])
    print('data not randomized')
    
trainSet.setTransform(transformSet)
testSet.setTransform(transformSet)


dataloader = DataLoader(trainSet, batch_size=mbSize,num_workers=numWorkers,shuffle=True,pin_memory=False)#,pin_memory=True)
testloader = DataLoader(testSet, batch_size=mbSize,shuffle=False, num_workers=numWorkers,pin_memory=False)
plt.plot(trainSet[0]['image'][5,:].to('cpu'))
plt.show()
plt.plot(testSet[0]['image'][5,:].to('cpu'))
plt.show()

#does NB refer to batch size? len of trainset divided mb (minibatch size?)
NB = np.floor(np.ceil(len(trainSet)/mbSize)/2)

model= simpleECG()
model = model.to(device)
#params = model.parameters()
criterion = nn.BCEWithLogitsLoss()


#optimizer = optim.SGD(params, lr=LR, momentum=0.9,weight_decay=WD)#2.0e-2)
#optimizer = AdaBound(params, lr=LR, final_lr=0.1,weight_decay=WD)#,eps=1e-4
if WEIGHT_DECAY==True:
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9,weight_decay=1e-4)
    print('optimizer = weight decay and SGD')
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=np.floor(0.33*Count), gamma=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.1,patience=np.floor(0.33*Count))#25

def test_model(model,criterion,device,testloader):
    model.eval()
    print('TESTING BEGINNING')
    with torch.no_grad():
        #roc_auc = []
        #roc_auc_ytrue_valid = []
        #roc_auc_ypred_valid = []
        all_labels = []
        all_outputs = []
        for i, data in enumerate(testloader,0):
            images, labels = data["image"], data["labels"]
            if torch.cuda.is_available():
                images, labels = data["image"].to(device),data["labels"].to(device)
            #roc_auc_ytrue.append(labels.long())
            #if doing np.random.rand
            #images = images.float()
            #forward pass
            outputs = model(images)
            #roc_auc_ypred.append(torch.sigmoid(outputs.float()))
            tx = (torch.sigmoid(outputs.float()))
            all_labels.append(labels.detach().long().cpu().numpy())
            all_outputs.append(tx.detach().cpu().numpy())
        #roc_ytrue_valid =  torch.cat(roc_auc_ytrue_valid,dim=0).detach().cpu()
        #roc_ypred_valid =  torch.cat(roc_auc_ypred_valid,dim=0).detach().cpu()
        #rocs = roc_auc_score(roc_ytrue_valid,roc_ypred_valid,multi_class='ovr',average='macro')
        all_labels = np.concatenate(all_labels,axis=0)
        all_outputs = np.concatenate(all_outputs,axis=0)
        
        ##EVALUATE ONLY ON 2 CLASSES
        all_outputs= all_outputs[:,:2]
        all_labels = all_labels[:,:2]
        
        print('Number of validation points: %i\n' %(all_outputs.shape[0]))
        print('Number of validation classes: %i\n' %(all_outputs.shape[1]))
        val_rocs = roc_auc_score(all_labels,all_outputs,multi_class='ovr',average='macro')
        print(f' VALIDATION Epoch rocs {val_rocs}')
    return val_rocs
        
def train_model(model,optimizer,criterion,scheduler,device,dataloader, testloader, n_epoch=EPOCH):
    model.train()
    for epoch in range(n_epoch):
        print('Epoch', epoch)
        train_loss = 0.0
        losses = []
        roc_auc = []
        roc_auc_ytrue = []
        roc_auc_ypred = []
        all_rocs = []

        
        for i, data in enumerate(dataloader,0):

            images, labels = data["image"], data["labels"]
            if torch.cuda.is_available():
                images, labels = data["image"].to(device),data["labels"].to(device)
            roc_auc_ytrue.append(labels.long())

            #if doing np.random.rand
            #images = images.float()

            #clear gradients
            optimizer.zero_grad()
            #forward pass
            outputs = model(images)
            #roc_auc_ypred.append(torch.sigmoid(outputs.float()))
            #find the loss
            loss = criterion(outputs,labels)
            #append losses to list
            losses.append(loss.item())
            #calculate gradients
            loss.backward()
            #update weights
            optimizer.step()
            with torch.no_grad():
                roc_auc_ypred.append(torch.sigmoid(outputs.float()))
            
            train_loss = loss.item()*images.size(0)
            if i % 100==0:
                print('Loss', i, train_loss)
        
        roc_ytrue =  torch.cat(roc_auc_ytrue,dim=0).detach().cpu()
        roc_ypred =  torch.cat(roc_auc_ypred,dim=0).detach().cpu()

        
        #EVALUATE ON 2 CLASSES ONLY
        roc_ytrue= np.array(roc_ytrue[:,:2])
        roc_ypred= np.array(roc_ypred[:,:2])
        

        train_rocs = roc_auc_score(roc_ytrue,roc_ypred,multi_class='ovr',average='macro')
        all_rocs.append(train_rocs)
        print(f' TRAINING Epoch rocs {train_rocs}')

        mean_loss = sum(losses)/len(losses)
        print(mean_loss)
        scheduler.step(mean_loss)
        print(f' Cost at epoch {epoch}, is {mean_loss}')
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        
    validation_loop = test_model(model,criterion,device,testloader)
    return model,train_rocs, validation_loop
   
model, train_rocs, val_rocs =train_model(model,optimizer,criterion,scheduler,device,dataloader,testloader, n_epoch=EPOCH)
# save reuslts to csv
columns_csv = ['DATE','PTH_NAME', 'NORMALIZATION', 'RANDOMIZE', 'FILTERED PTS', 'WEIGHT_DECAY', 'TWO_CLASS','TRAIN_ACC', 'VAL_ACC']      
# The data assigned to the dictionary 


dict={ 'DATE': todays_date, 'PTH_NAME': SAVE_PTH_NAME, 'NORMALIZATION': NORMALIZATION, 'RANDOMIZE': RANDOMIZE, 'FILTERED PTS': FILTER_PTS,
     'WEIGHT_DECAY': WEIGHT_DECAY, 'TWO_CLASS': TWO_CLASS,'TRAIN_ACC': train_rocs, 'VAL_ACC': val_rocs }

with open('ptbxl_results_may32022.csv', 'a', newline='') as f_object:
    # Pass the CSV  file object to the Dictwriter() function
    # Result - a DictWriter object
    dictwriter_object = DictWriter(f_object, fieldnames=columns_csv)
    # Pass the data in the dictionary as an argument into the writerow() function
    dictwriter_object.writerow(dict)
    # Close the file object
    f_object.close()

MODEL_SAVE = '/cluster/home/cathy.ongly/data/'+ SAVE_PTH_NAME + '.pth'
#MODEL_SAVE = '/cluster/home/cathy.ongly/data/ptb-xl_rand_sept16.pth'
torch.save(model.state_dict(), MODEL_SAVE)
