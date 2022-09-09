import imp
from locale import normalize
from random import shuffle
import cv2
import numpy as np
from torch.utils.data import Dataset,DataLoader
from engine import utils
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image


class Randomize(object):
    """Shuffles pixels of given image
    Different shuffle applied for each channel
    """
    def __init__(self, dims,p=1.0,shuffle_channels_independently=False):
        self.dims = dims
        self.p = p
        self.shuffle_channels_independently =  shuffle_channels_independently

    def __call__(self, img):
        if np.random.rand(1)[0] <= self.p:
            shp = img.shape
            idx = np.random.permutation(np.arange(np.prod(shp[1:])))
            for i in list(self.dims):
                if self.shuffle_channels_independently:
                    idx = np.random.permutation(np.arange(np.prod(shp[1:])))
                img[i,...] = img[i,...].view(-1)[idx].view(shp[1:])
        return img

class ShapesDataset(Dataset):
    def __init__(self,num_samples=1000,img_sz=256,mu=0,sigma=1,tfms=None):
        self.num_samples = 1000
        self.img_sz = img_sz
        self.tfms = tfms
        self.mu = mu
        self.sigma = sigma

        self.labels = np.random.randint(low=0,high=2,size=num_samples)
        self.labels = np.eye(2)[self.labels]
        self.colors = [(145, 145, 145),]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        label = self.labels[idx]
        cl = np.argmax(label)
        img = np.zeros((self.img_sz,self.img_sz,3),dtype=np.uint8)+128
        xy = np.random.randint(low=int(self.img_sz*.20),high=int(self.img_sz*.80),size=2)
        r = np.random.randint(low=int(self.img_sz*.10),high=int(self.img_sz*.25),size=1)[0]

        if cl == 0:
            x,y = xy[:]
            cv2.circle(img, (x,y), int(r/1.25), self.colors[0], thickness=2)

        else:
            x1,y1 = xy[:]
            l = r
            x2,y2 = x1+l,y1+l
            cv2.rectangle(img, (x1,y1), (x2,y2), self.colors[0], thickness=2)

        if self.mu and self.sigma:
            mu,sigma = self.mu[cl],self.sigma[cl]
            mu = mu + np.random.normal(0,6)
            gaussian = np.random.normal(mu, sigma, (self.img_sz,self.img_sz)) 
            for ch in [0,1,2]:
                img[:, :, ch] = img[:, :, ch] + gaussian

        img = Image.fromarray(img)

        if self.tfms:
            img = self.tfms(img)
        return img,label

def main():
    with utils.set_seed(42):
        
        tfms = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            # Randomize(dims=(0,1,2),p=1,shuffle_channels_independently=False),
        ])

        ds = ShapesDataset(num_samples=250,mu=[100,95],sigma=[5,5],tfms=tfms)
        dl = DataLoader(ds,batch_size=8,shuffle=True,num_workers=2)

        for idx,(x,y) in enumerate(dl):
            print(idx,x.shape,y.shape,utils.show_tensor_stats(x))
            path = utils.get_prepared_path('example_code/circles_sqr/%s.png'%(str(idx).zfill(5)))
            save_image(x,path,normalize=True,scale=False)
            print('Samples saved to %s'%path)
            if idx == 0:
                break



