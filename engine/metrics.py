from sklearn import metrics
import numpy as np
import torch


class SelectedClassAUROC():
    def __init__(self,cl_idx=[2,5,6,8,10],num_classes=14):
        self.cl_idx = np.array(cl_idx)
        self.num_classes = num_classes
        self.y_scores = None
        self.y_true = None

    def update(self,op,y):
        self.y_scores = torch.cat([self.y_scores,op]) if self.y_scores is not None else op
        self.y_true = torch.cat([self.y_true,y]) if self.y_true is not None else y

    def convert_to_numpy(self,arr):
        if isinstance(arr,torch.Tensor):
            if arr.is_cuda:
                arr = arr.cpu()
            arr = arr.numpy()
        else:
            arr = np.array(arr)
        return arr

    def compute(self):
        self.y_scores = self.convert_to_numpy(self.y_scores) 
        self.y_true = self.convert_to_numpy(self.y_true) 
        aucs = []
        for idx in self.cl_idx:
            auc = metrics.roc_auc_score(self.y_true[:,idx],self.y_scores[:,idx])
            aucs.append(auc)
        aucs = np.array(aucs)
        avg_auc = np.nanmean(aucs)
        return avg_auc,aucs

    def get_stored_values(self,astype='tensor'):
        if astype == 'tensor':
            return torch.Tensor(self.y_scores),torch.Tensor(self.y_true)
        else:    
            return self.y_scores,self.y_true
