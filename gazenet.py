import os 
import numpy as np 
from numpy.linalg import norm,inv
import cv2 
from collections import OrderedDict
import torch 
import torch.nn as nn 
from torch.nn import SmoothL1Loss
from torch.nn import init 
import torch.nn.functional as F 
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader 
from imgaug import augmenters as iaa
import gc 
import time 
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from resnext import *
class GazeNet(nn.Module):
    """
    The end_to_end model of Gaze Prediction Training
    """
    def __init__(self):
        super(GazeNet, self).__init__()
        self.face_net = resnext50(4,32)
        self.leye_net = resnext50(4,32)
        self.reye_net = resnext50(4,32)

    def calc_gaze_lola(self,head,eye):
        head_lo = head[:,0]
        head_la = head[:,1]
        eye_lo = eye[:,0]
        eye_la = eye[:,1]
        cA = torch.cos(head_lo/180*np.pi)
        sA = torch.sin(head_lo/180*np.pi)
        cB = torch.cos(head_la/180*np.pi)
        sB = torch.sin(head_la/180*np.pi)
        cC = torch.cos(eye_lo/180*np.pi)
        sC = torch.sin(eye_lo/180*np.pi)
        cD = torch.cos(eye_la/180*np.pi)
        sD = torch.sin(eye_la/180*np.pi)
        g_x = - cA * sC * cD + sA * sB * sD - sA * cB * cC * cD
        g_y = cB * sD + sB * cC * cD
        g_z = sA * sC * cD + cA * sB * sD - cA * cB * cC * cD
        gaze_lo = torch.atan2(-g_x,-g_z)*180.0/np.pi
        gaze_la = torch.asin(g_y)*180.0/np.pi
        gaze_lo = gaze_lo.unsqueeze(1)
        gaze_la = gaze_la.unsqueeze(1)
        gaze_lola = torch.cat((gaze_lo,gaze_la),1)
        return gaze_lola
    def forward(self,img_face,img_leye,img_reye):
        return_dict = {}
        head = self.face_net(img_face)
        leye = self.leye_net(img_leye)
        reye = self.reye_net(img_reye)
        eye = (leye + reye) / 2
        #print("head",head.shape)
        #print("eye",eye.shape)
        gaze_lola = self.calc_gaze_lola(head,eye)
        #对头部，眼睛和视线朝向的角度做归一化到[0,1]范围内
        head = (head + 90)/180
        eye = (eye + 90)/180
        gaze_lola = (gaze_lola + 90)/180
        return_dict['head'] = head
        return_dict['eye'] = eye
        return_dict['gaze'] = gaze_lola
        return return_dict