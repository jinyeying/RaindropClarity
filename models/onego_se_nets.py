import torch
import torchvision
import math
import torch.nn as nn
from torchvision.models import ResNet
import numpy as np
import random
import itertools
import skimage as ski
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
from PIL import Image
from scipy import ndimage
from scipy.special import gamma
from skimage.transform import warp
import cv2
import h5py

class InsNorm(nn.Module):    
    def __init__(self, dim, eps=1e-9):
        super(InsNorm, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()
        
    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x):        
        flat_len = x.size(2)*x.size(3)
        vec = x.view(x.size(0), x.size(1), flat_len)
        mean = torch.mean(vec, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        var = torch.var(vec, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((flat_len - 1)/float(flat_len))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var+self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out
        
        
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False 

        
class SELayer(nn.Module):
    def __init__(self, channel, reduction=64):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, reduction),
                nn.ReLU(inplace=True),
                nn.Linear(reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y    
        

def CropSample(im_input, label, crop_size):
    if isinstance(label, np.ndarray):
        label = Image.fromarray(label)
    if isinstance(im_input, np.ndarray):
        im_input = Image.fromarray(im_input)

    W, H = label.size
    x_offset = random.randint(0, W - crop_size)
    y_offset = random.randint(0, H - crop_size)
    label    = label.crop((x_offset, y_offset,
                           x_offset+crop_size, y_offset+crop_size))
    im_input = im_input.crop((x_offset, y_offset,
                              x_offset+crop_size, y_offset+crop_size))
    return im_input, label
    

def DataAugmentation(im_input, label):
    if random.random() > 0.5:
        label    = label.transpose(   Image.FLIP_LEFT_RIGHT)
        im_input = im_input.transpose(Image.FLIP_LEFT_RIGHT)
    return im_input, label


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# SE-ResNet Module    
class SEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=64, with_norm=False):
        super(SEBasicBlock, self).__init__()
        self.with_norm = with_norm
        
        self.conv1 = conv3x3(inplanes, planes, stride)                    
        self.conv2 = conv3x3(planes, planes, 1)        
        self.se = SELayer(planes, reduction)
        self.relu = nn.ReLU(inplace=True)        
        if self.with_norm:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        if self.with_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.with_norm:
            out = self.bn2(out)
        out = self.se(out)        
        out += x        
        out = self.relu(out)
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        self.softmax  = torch.nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out



class PAM_Module(nn.Module):
    """ Position attention module"""
    #paper: Dual Attention Network for Scene Segmentation
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//16, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//16, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature ( B X C X H X W)
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        import pdb; pdb.set_trace()
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out




class MFAttention(nn.Module):
    def __init__(self, channels, r):
        super(MFAttention, self).__init__()
        inter_channels = int(channels // r)
        
        self.local_conv1 = nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.local_bn1   = nn.BatchNorm2d(inter_channels)
        self.local_relu1 = nn.ReLU(inplace=True)
        self.local_conv2 = nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0)
        self.local_bn2   = nn.BatchNorm2d(channels)

        self.global_gap   = nn.AdaptiveAvgPool2d(1)
        self.global_conv1 = nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.global_bn1   = nn.BatchNorm2d(inter_channels)
        self.global_relu1 = nn.ReLU(inplace=True)
        self.global_conv2 = nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0)
        self.global_bn2   = nn.BatchNorm2d(channels)

        self.sigmod       = nn.Sigmoid()

    def forward(self, x, residual):
        xl = self.local_conv1(x)
        xl = self.local_bn1(xl)
        xl = self.local_relu1(xl)
        xl = self.local_conv2(xl)
        xl = self.local_bn2(xl)

        xg = self.global_gap(x)
        xg = self.global_conv1(xg)
        xg = self.global_bn1(xg)
        xg = self.global_relu1(xg)
        xg = self.global_conv2(xg)
        xg = self.global_bn2(xg)

        xlg = xl + xg
        wei = self.sigmod(xlg)

        out = x * wei + residual *(1-wei)

        return out