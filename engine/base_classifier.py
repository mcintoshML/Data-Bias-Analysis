import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from networks import arch
from dataloaders.radiology_2d import RadiologyDataset2D
from dataloaders.ild_3d import RadiologyDataset3D
from dataloaders.audio2img import Audio2Image
from dataloaders.ecg_1d import ECG_Dataset
from engine.metrics import SelectedClassAUROC
from engine import utils
from constants import MODEL_DIR


class SupervisedClassifier():
    def __init__(self, args):
        self.args = args
        self.name = '%s_seed%d' % (args.name, args.seed)

    def get_path(self, item):
        path = '%s%s/%s' % (MODEL_DIR,self.name, item)
        utils.create_folder(path)
        return path

    def load_model(self, resume=None):
        args = self.args
        self.model = arch.get_arch(arch=args.arch,pretrained=args.pretrained,num_classes=args.num_classes,num_channels=args.num_channels)
        self.model = nn.DataParallel(self.model).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr,betas = args.betas)
        self.early_stop = args.early_stop
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.1, patience=np.floor(0.33*self.early_stop), verbose=1)
        self.best_valid_auc = 0
        self.no_improv = 0
        self.global_step = 0
        self.epoch = 0
        self.sup_loss = nn.BCEWithLogitsLoss()
        self.writer = SummaryWriter(log_dir=self.get_path('logs/'))
        if resume in ['best', 'last']:
            path = self.get_path('%s.pth' % resume)
            if not os.path.exists(path):
                print('No saved model at %s. Starting with new model'%path)
                return
            print('Resuming from %s'%resume)
            state = torch.load(path,map_location=self.device)
            self.model.load_state_dict(state['model_state_dict'])
            self.opt.load_state_dict(state['opt_state_dict'])
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
            self.early_stop = state['early_stop']
            self.best_valid_auc = state['best_valid_auc']
            self.no_improv = state['no_improv']
            self.global_step = state['global_step']
            self.epoch = state['epoch']

    def save_model(self, valid_auc):
        
        if valid_auc > self.best_valid_auc:
            tqdm.write('Best model updated. AUC changed from %.6f to %.6f at epoch %d' % (
                self.best_valid_auc, valid_auc, self.epoch))
            self.best_valid_auc = valid_auc
            self.no_improv = 0
            best_model = True
        else:
            self.no_improv += 1
            best_model = False

        state = {
            'model_state_dict': self.model.state_dict(),
            'opt_state_dict': self.opt.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'early_stop': self.early_stop,
            'best_valid_auc': self.best_valid_auc,
            'no_improv': self.no_improv,
            'global_step': self.global_step,
            'epoch': self.epoch,
        }
        if best_model:
            torch.save(state, self.get_path('best.pth'))
        torch.save(state, self.get_path('last.pth'))

    def get_dataloader(self, which):
        ds_class = {'ild':RadiologyDataset3D,'rad': RadiologyDataset2D,'audio2img':Audio2Image,'ecg':ECG_Dataset}[self.args.ds_class]
        dataset = ds_class(which=which,shuffle_pixels=self.args.shuffle_pixels,
            preload_to_ram=self.args.preload_to_ram,num_samples=self.args.num_samples,
            folds=self.args.folds,
            disable_tqdm=self.args.disable_tqdm)        
        args = self.args
        shuffle = 'train' in which 
        dl = DataLoader(dataset, batch_size=args.batch_size,num_workers=args.num_workers, shuffle=shuffle)
        return dl

    def validate(self,dl,save_path=None):
        self.model.eval()
        metric = SelectedClassAUROC(cl_idx=self.args.eval_classes, num_classes=self.args.num_classes)
        with torch.no_grad():
            for x, y in tqdm(dl, leave=False,disable=self.args.disable_tqdm,desc='Validate'):
                x = x.to(self.device).float()
                y = y.to(self.device).float()
                op = self.model(x)
                metric.update(op, y)
            avg_auc, aucs = metric.compute()
            op, y = metric.get_stored_values()
            xe_loss = self.sup_loss(op, y).item()
            if save_path is not None:
                utils.create_folder(save_path)
                np.savez_compressed(save_path,y_scores=op,y_true=y)
                print('Saved to %s'%save_path,op.shape,y.shape)
        return {'avg_auc': avg_auc, 'xe': xe_loss, 'aucs': aucs}

    def test(self, test_on='chexpert_valid', replace=True):
        path = self.get_path('res_%s.csv' %(test_on))
        if not replace and os.path.exists(path):
            df = pd.read_csv(path)
            print(df)
        else:
            args = self.args
            self.load_model('best')
            dl = self.get_dataloader(which=test_on)
            if args.test_eval_classes!='same':
                self.args.eval_classes = args.test_eval_classes
            results = self.validate(dl,save_path=self.get_path('res_ops/%s.npz'%(test_on)))
            data = np.array([args.exp, args.train_on, args.valid_on, test_on, args.shuffle_pixels, results['avg_auc']])
            df = pd.Series(data, index=['exp', 'train_on', 'valid_on', 'test_on','shuffle_pixels','avg_auc'])
            df.to_csv(path)
            print(df)                

def main():
    clf = SupervisedClassifier()
