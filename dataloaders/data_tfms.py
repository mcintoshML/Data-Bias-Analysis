import numpy as np
import torch


class Randomize(object):
    """Shuffles pixels of given image
    Different shuffle applied for each channel
    """
    def __init__(self, dims,p=1.0):
        self.dims = dims
        self.p = p

    def __call__(self, img):
        if np.random.rand(1)[0] <= self.p:
            shp = img.shape
            for i in list(self.dims):
                idx = np.random.permutation(np.arange(np.prod(shp[1:])))
                img[i,...] = img[i,...].reshape(-1)[idx].reshape(shp[1:])
        return img

class Randomize2(object):
    """Shuffles pixels of given image
    Same shuffle applied for each channel
    """
    def __init__(self, dims,p=1.0):
        self.dims = dims
        self.p = p

    def __call__(self, img):
        if np.random.rand(1)[0] <= self.p:
            shp = img.shape
            # Same shuffle is applied to all 3 channels in RGB so that pixels are shuffled 
            idx = np.random.permutation(np.arange(np.prod(shp[1:])))
            for i in list(self.dims):
                img[i,...] = img[i,...].reshape(-1)[idx].reshape(shp[1:])
        return img

class ToTensor():
    def __init__(self,dtype=torch.float32):
        self.dtype = dtype
    
    def __call__(self,x):
        return torch.Tensor(x).to(self.dtype)

class Noop():
    def __init__(self):
        pass
        
    def __call__(self,x):
        return x        