import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from engine import utils
from engine.arg_handler import get_combined_args
from engine.base_classifier import SupervisedClassifier


class ExperimentRunner(SupervisedClassifier):
    def __init__(self, args):
        args.name = '%s_%s' % (args.train_on, args.valid_on)
        if args.shuffle_pixels:
            args.name = '%s_shuffled' % (args.name)

        super().__init__(args)
        self.device = torch.device(args.cuda_device if torch.cuda.is_available() else 'cpu')

    def validate_and_checkpoint(self, valid_dl,use_scheduler=False):
        self.model.eval()
        valid_metrics = self.validate(valid_dl)
        if self.args.tensor_log:
            self.writer.add_scalar('valid_loss', valid_metrics['xe'], self.global_step)
            self.writer.add_scalar('valid_avg_auc', valid_metrics['avg_auc'], self.global_step)
        if use_scheduler:
            self.scheduler.step(valid_metrics['xe'])
        self.save_model(valid_metrics['avg_auc'])
        tqdm.write('[%d/%s] auc %.4f xe %.4f' % (self.epoch,
                   self.global_step, valid_metrics['avg_auc'], valid_metrics['xe']))
        self.model.train()
        return valid_metrics

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

    def train(self, resume=None,num_epochs=None):
        with open(self.get_path('hyperparams.txt'),'w') as file:
            print(self.args,file=file)

        self.load_model(resume=resume)
        args = self.args
        train_dl = self.get_dataloader(which=args.train_on)
        valid_dl = self.get_dataloader(which=args.valid_on)

        if resume and num_epochs:
            args.num_epochs = num_epochs

        for e in tqdm(np.arange(self.epoch, args.num_epochs),initial=self.epoch,disable=self.args.disable_tqdm,desc='Training'):
            self.epoch = e
            valid_metrics = self.train_epoch(e,train_dl, valid_dl)

            if self.no_improv > self.early_stop:
                print('Early Stopping @ %d epoch' % e)
                break
        self.writer.close()

def main():

    args = get_combined_args()
    with utils.set_seed(args.seed):
        exp = ExperimentRunner(args)
        if args.task == 'train':
            exp.train(resume='new')
        elif args.task == 'resume':
            exp.train(resume='last',num_epochs=50)
        elif args.task == 'eval':
            exp.test(test_on=args.test_on, replace=True)
        elif args.task == 'check_and_eval':
            exp.test(test_on=args.test_on, replace=False)
