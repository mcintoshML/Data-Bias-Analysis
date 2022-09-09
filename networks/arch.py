from torchvision.models import densenet121
import torch.nn as nn
import torch
from torchvision.models import resnet34
from networks import vgg3d
from networks import ecg1d


def get_arch(arch,pretrained=False,num_classes=14,num_channels=1):
    if arch == 'dn121':
        net = densenet121(pretrained=pretrained)
        net.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        net.classifier = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
    elif arch == 'vgg3d':
        cfg = [4, 'M', 8, 'M', 16, 32, 'M', 32,32, 'M', 64, 64, 'M']
        net = vgg3d._vgg(arch='vgg', cfg=cfg, batch_norm=False, pretrained=False,c3D=True, progress=False,in_channels=num_channels)
        net.classifier = nn.Linear(in_features=64, out_features=num_classes, bias=True)
    elif arch == 'resnet34':
        net = resnet34(pretrained=pretrained)
        num_ftrs = net.fc.in_features
        net.fc = nn.Identity()
        net.fc = nn.Linear(num_ftrs,num_classes,bias=True)
    elif arch == 'ecg1d':
        assert pretrained == False
        assert num_channels == 12
        net = ecg1d.simpleECG(num_classes=num_classes,num_channels=num_channels)
    return net


def test_dn121():
    net = get_arch(arch='dn121')
    print(net)

    z = torch.randn(5,1,320,320)
    op = net(z)
    print(op.shape)

def test_vgg3d():
    net = get_arch(arch='vgg3d')
    print(net)

    z = torch.randn(5,1,256,256,128)
    op = net(z)
    print(op.shape)

def test_resnet34():
    net = get_arch(arch='resnet34')
    print(net)

    z = torch.randn(5,3,224,224)
    op = net(z)
    print(op.shape)

def test_ecg1d():
    net = get_arch(arch='ecg1d',num_classes=5,num_channels=12)
    print(net)

    z = torch.randn(5,12,1000)
    op = net(z)
    print(op.shape)    



def main():
    # test_dn121()
    # test_vgg3d()
    # test_resnet34()
    test_ecg1d()