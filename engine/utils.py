from pathlib import Path
import os
import shutil
import contextlib
import joblib
from tqdm import tqdm    
import numpy as np
from joblib import delayed, Parallel, parallel_backend
import multiprocessing
import torch
from scipy import special
import matplotlib.pyplot as plt


def create_folder(path,clear=False):
    if '.' in os.path.basename(path):
        path = os.path.dirname(path)
    if clear:
        if os.path.exists(path):
            shutil.rmtree(path)
    Path(path).mkdir(parents=True, exist_ok=True)

def get_prepared_path(path):
    create_folder(path)
    return path

def unique(arr,min_count=0):
    values,counts = np.unique(arr,return_counts=True)
    for val,count in zip(values,counts):
        if count>min_count:
            print('%s - %d'%(val,count))

def check_paths(paths):
    paths = np.array(paths)
    idx = Parallel(n_jobs=-1,prefer='threads')(delayed(os.path.exists)(path) for path in tqdm(paths))
    idx = np.array(idx)
    if np.sum(idx)!=len(paths):
        print('Invalid Paths')
        print(paths[idx==False])
    unique(idx)

def show_tensor_stats(arr):
    return '[%.6f to %.6f] Mean %.6f Std %.6f'%(torch.min(arr),torch.max(arr),torch.mean(arr),torch.std(arr))

def show_numpy_stats(arr):
    return '[%.6f to %.6f] Mean %.6f Std %.6f'%(np.min(arr),np.max(arr),np.mean(arr),np.std(arr))

@contextlib.contextmanager
def set_seed(seed):
    """
        Usage
            with utils.set_seed(args.seed):
                exp = ExperimentRunner(args)
                exp.train(resume='new')    
    """
    state = np.random.get_state()
    torch_state = torch.get_rng_state()
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    try:
        yield
    finally:
        np.random.set_state(state)
        torch.set_rng_state(torch_state)


def visualize_dl(dl,num_batches=5,folder='vis_imgs/dl/'):
    import torchvision
    for idx,(x,y) in enumerate(dl):
        print(idx,x.shape,y.shape,show_tensor_stats(x))
        path = '%s%s.jpg'%(folder,str(idx).zfill(10))
        create_folder(path)
        torchvision.utils.save_image(x,path,normalize=True)
        if idx == num_batches:
            break




import sys
def print_as_error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs) 

def softmax(x,axis=1):
    if isinstance(x,torch.Tensor):
        return torch.softmax(x,dim=axis)
    if isinstance(x,np.ndarray):   
        return special.softmax(x,axis=axis)

def sigmoid(x):
    if isinstance(x,torch.Tensor):
        return torch.sigmoid(x)
    if isinstance(x,np.ndarray):
        return special.expit(x)
   

def mean_confidence_interval(data,confidence=.95):
    # https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/
    alpha = confidence
    stats = data

    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    m = np.mean(data)

    return m,lower,upper   

def save_plot(path,dpi=150):
    create_folder(path)
    plt.savefig(path,dpi=dpi,bbox_inches='tight')
    plt.close()

def calculate_running_mean_std(dataloader,batch_axis=0,verbose=1):
    # https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data,__ in tqdm(dataloader):
        batch_samples = data.size(batch_axis)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    if verbose>0:
        print(mean,std)
    return mean,std