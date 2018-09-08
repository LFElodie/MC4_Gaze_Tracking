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


from gazedataset import *
from gazenet import *
def get_test_loader(data_dir,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=False):
    dataset = GazeDataset(data_dir,"test")
    data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=shuffle,
    num_workers=num_workers, pin_memory=pin_memory,
    ) 
    return data_loader
def output_predict(dataloader,output_path,pretrained_model = None):
    #test mode
    model = GazeNet()
    if pretrained_model:
        pt = torch.load(pretrained_model)
        model.load_state_dict(pt["model"])
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"#use one gpu for testing?
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    dataiterator = iter(dataloader)
    pred = {}
    while True:
        try:
            input_data = next(dataiterator)
        except StopIteration:
            break
        for key in input_data:
            if key!= "img_name":
                if use_cuda:
                    input_data[key] = Variable(input_data[key]).type(torch.cuda.FloatTensor)
                else:
                    input_data[key] = Variable(input_data[key]).type(torch.FloatTensor)
        img_head = input_data['head_image']
        img_leye = input_data['leye_image']
        img_reye = input_data['reye_image']

        output = model(img_head,img_leye,img_reye)
        if output is None:
            raise Exception
        gaze_lola = output["gaze"].data.cpu().numpy()
        gaze_lola = gaze_lola*180 - 90
        img_name_batch = input_data['img_name']
        for idx,img_name in enumerate(img_name_batch):
            pred[img_name] = gaze_lola[idx]
    with open(output_path,"w") as f:
        for k,v in pred.items():
            f.write(k.split(".")[0]+"\n")
            f.write("%0.3f\n" % v[0])
            f.write("%0.3f\n" % v[1])


if __name__ == "__main__":
    test_data_dir = "/data/mc_data/MC4/test"
    output_path = "./pred.txt"
    model_path = "./ckpt/0.01/1_epoch.pth"
    test_loader = get_test_loader(test_data_dir)
    output_predict(test_loader,output_path,model_path)
