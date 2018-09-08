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



def image_normalize(im_data,transform=None):
    im_data = cv2.resize(im_data,(224,224))
    if transform:
        im_data = transform.augment_image(im_data)
    im_data = im_data[np.newaxis,:].astype(np.float64)
    return im_data


class GazeDataset(Dataset):
    def __init__(self, data_dir, mode, transform=None):
        """
        Args:
        data_dir (string): Directory with all the images.
        mode (string): train/val/test subdirs.
        transform (callable, optional): Optional transform to be applied
        on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.img_list = os.listdir(os.path.join(data_dir, "head"))
        if self.mode == "train":
            self.head_label = self.load_gt(
                os.path.join(data_dir, "head_label.txt"))
            self.gaze_label = self.load_gt(
                os.path.join(data_dir, "gaze_label.txt"))
            self.eye_label = self.load_gt(
                os.path.join(data_dir, "eye_label.txt"))

    def load_kp(self, filename):
        ret = {}
        with open(filename, 'r') as kpfile:
            while True:
                line = kpfile.readline()
                if not line:
                    break
                img_filename = line.strip("\n")
                #im_data = cv2.imread(os.path.join(p,str(i),img_filename))
                src_point = []
                line = kpfile.readline()
                p_count = int(line.strip("\n"))
                for j in range(p_count):
                    x = float(kpfile.readline().strip("\n"))
                    y = float(kpfile.readline().strip("\n"))
                    src_point.append((x, y))
                ret[img_filename] = src_point
        return ret

    def load_gt(self, filename):
        ret = {}
        with open(filename, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip("\n") + ".png"
                lo = float(f.readline().strip("\n"))
                la = float(f.readline().strip("\n"))
                ret[line] = np.array([lo, la], dtype=np.float32)
        return ret

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        head_image = cv2.imread(os.path.join(
            self.data_dir, "head", self.img_list[idx]), cv2.IMREAD_GRAYSCALE)
        # 头部图像还包含了大量背景区域，需要做居中裁剪
        mid_x, mid_y = head_image.shape[0] // 2, head_image.shape[1] // 2
        head_image = head_image[mid_x - 112:mid_x + 112, mid_y - 112:mid_y + 112]
        leye_image = cv2.imread(os.path.join(
            self.data_dir, "l_eye", self.img_list[idx]), cv2.IMREAD_GRAYSCALE)
        reye_image = cv2.imread(os.path.join(
            self.data_dir, "r_eye", self.img_list[idx]), cv2.IMREAD_GRAYSCALE)
        # eye_image = leye_image if np.random.rand() < 0.5 else reye_image
        head_image = image_normalize(head_image)
        # eye_image = image_normalize(eye_image)

        leye_image = image_normalize(leye_image)
        reye_image = image_normalize(reye_image)

        if self.mode == "train":
            head_lola = self.head_label[self.img_list[idx]]
            eye_lola = self.eye_label[self.img_list[idx]]
            gaze_lola = self.gaze_label[self.img_list[idx]]
            # sample = {'img_name': self.img_list[idx], 
            #             'head_image': head_image, 'eye_image': eye_image, 'head_lola': head_lola, 'eye_lola': eye_lola, 'gaze_lola': gaze_lola}
            sample = {'img_name': self.img_list[idx], 
                        'head_image': head_image, 'leye_image': leye_image, 'reye_image': reye_image, 'head_lola': head_lola, 'eye_lola': eye_lola, 'gaze_lola': gaze_lola}
        else:
            # sample = {'img_name': self.img_list[idx],
            #           'head_image': head_image, 'eye_image': eye_image}
            sample = {'img_name': self.img_list[idx],
                      'head_image': head_image, 'leye_image': leye_image, 'reye_image': reye_image}
        return sample
