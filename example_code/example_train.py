from torchvision.models import densenet121
from example_code.example_dataset import ShapesDataset,Randomize
from torchvision import transforms
from engine.sup_classifier import ExperimentRunner
from engine import utils 
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader
import os
from tqdm import tqdm
from engine import arg_handler
import numpy as np


class Net(nn.Module):
    def __init__(self,pretrained=False,num_classes=2):
        super().__init__()
        self.net = densenet121(pretrained=pretrained)
        self.net.classifier = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

    def forward(self,x):
        return self.net(x)

class Supervised_Example(ExperimentRunner):
    def __init__(self, args):
        super().__init__(args)
        print(self.name)

    def load_model(self, resume=None):
        args = self.args
        self.model = Net(num_classes=2,pretrained=False)
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

    def get_dataloader(self, which):
        args = self.args
        if args.shuffle_pixels:
            tfms = transforms.Compose([
                transforms.Resize(128),
                transforms.ToTensor(),
                Randomize(dims=(0,1,2),p=1,shuffle_channels_independently=False),
            ])
        else:
            tfms = transforms.Compose([
                transforms.Resize(128),
                transforms.ToTensor(),
            ])            


        if 'CirSqr_train' == which:
            dataset = ShapesDataset(num_samples=2000,mu=[100,95],sigma=[5,5],tfms=tfms)
        elif 'CirSqr_valid' == which:
            dataset = ShapesDataset(num_samples=1001,mu=[100,95],sigma=[5,5],tfms=tfms)
        elif 'CirSqrExt_valid' == which:
            dataset = ShapesDataset(num_samples=1000,mu=[100,100],sigma=[5,5],tfms=tfms)

        shuffle = '_train' in which 
        dl = DataLoader(dataset, batch_size=args.batch_size,num_workers=args.num_workers, shuffle=shuffle)
        return dl

    def train_epoch(self, e, train_dl, valid_dl):
        self.model.train()
        for x, y in tqdm(train_dl, leave=False, disable=self.args.disable_tqdm,desc='Train Epoch %d'%e):
            self.opt.zero_grad()
            x = x.to(self.device).float()
            y = y.to(self.device).float()
            op = self.model(x)
            loss = self.sup_loss(op, y)
            if self.args.tensor_log:
                self.writer.add_scalar('train_loss', loss, self.global_step)
            loss.backward()
            self.opt.step()
            self.global_step += 1

            if self.global_step > 0 and self.args.chkpt_every>0 and self.global_step % self.args.chkpt_every == 0:
                self.validate_and_checkpoint(valid_dl,use_scheduler=False)

        self.validate_and_checkpoint(valid_dl,use_scheduler=True)

    

def main():

    expt_args = {

    'name':'example',
    'exp':'example',
    'seed':20,
    'num_classes':14,
    'pretrained':False,
    
    'lr':1e-4,
    'betas':(0.9, 0.999),
    'num_epochs':1,
    'tensor_log':False,
    'chkpt_every':4800,
    'early_stop':10,

    'shuffle_pixels': False, 

    'train_on':'CirSqr_train',
    'valid_on':'CirSqr_valid',
    'test_on':'CirSqrExt_valid',
    'folds':None,
    
    'task':'eval',
    'disable_tqdm':False,
    'ngpu':1,

    'eval_classes':[0,1],
    'test_eval_classes':'same',
    # 'batch_size':16,
    'batch_size':32,
    'num_workers':4,
    'img_sz':256,
    'cuda_device':'cuda:0',

    } 
    
    args = arg_handler.get_combined_args(expt_args)
    for shuffle_pixels in (False,True):
        args.shuffle_pixels = shuffle_pixels
        with utils.set_seed(args.seed):
            exp = Supervised_Example(args)
            if args.task == 'train':
                exp.train(resume='new')
            if args.task in ('train','eval'):
                exp.test(test_on=args.valid_on)
                exp.test(test_on=args.test_on)
