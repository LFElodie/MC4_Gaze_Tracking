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
def get_train_valid_loader(dataset,
                            batch_size,
                            seed = 0,
                            valid_size=0.1,
                            shuffle=True,
                            show_sample=False,
                            num_workers=8,
                            pin_memory=False):
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers)

    return (train_loader, valid_loader)



def main(dataset,pretrained_model):
    os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"#8 gpus per node
    use_cuda = torch.cuda.is_available()
    model = GazeNet()
    if pretrained_model:
        pt = torch.load(pretrained_model)
        model.load_state_dict(pt["model"])
        print('load pretrained model success')
    if use_cuda:
        model = nn.DataParallel(model).cuda()
    GPU_COUNT = torch.cuda.device_count()
    epoch = 0#one epoch is to iterate over the entire training set
    steps = 200000
    print("Training Starts!")
    phase = "train"
    train_loader,valid_loader = get_train_valid_loader(dataset,8*GPU_COUNT)
    dataiterator = iter(train_loader)
    start_time = time.time()
    for step in range(steps):
        if phase == "train":
            #train mode
            #define optimizer
            if epoch == 0:
                learning_rate = 0.01
                optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=5e-4)
            elif epoch == 5:
                learning_rate = 0.001
                optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=5e-4)
            elif epoch == 10:
                learning_rate = 0.0005
                optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=5e-4)
            elif epoch == 15:
                learning_rate = 0.0001
                optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=5e-4)
            optimizer.zero_grad()
            try:
                input_data = next(dataiterator)
            except StopIteration:
                phase = "val"
                continue
            for key in input_data:
                if key!="img_name":
                    if use_cuda:
                        input_data[key] = Variable(input_data[key]).type(torch.cuda.FloatTensor)
                    else:
                        input_data[key] = Variable(input_data[key]).type(torch.FloatTensor)
            img_head = input_data['head_image']
            img_leye = input_data['leye_image']
            img_reye = input_data['reye_image']
            head_gt = input_data['head_lola']
            gaze_gt = input_data['gaze_lola']
            eye_gt = input_data['eye_lola']
            #将真值也做归一化，和模型输出量纲相同
            head_gt = (head_gt + 90)/180
            gaze_gt = (gaze_gt + 90)/180
            eye_gt = (eye_gt + 90)/180
            output = model(img_head,img_leye,img_reye)
            loss_fn = SmoothL1Loss()
            loss_head = loss_fn(output['head'],head_gt)
            loss_eye = loss_fn(output['eye'],eye_gt)
            loss_gaze = loss_fn(output['gaze'],gaze_gt)
            total_loss = loss_head + loss_eye + loss_gaze
            if (step+1)%100==0:
                print("Step: {} Elapsed Time: {}s".format(step+1,time.time()-start_time))
                print("head: {:.5f} eye: {:.5f} gaze: {:.5f} total: {:.5f}"\
                    .format(loss_head.item(),loss_eye.item(),loss_gaze.item(),total_loss.item()))
            if (step+1)%1000==0:
                if not os.path.exists("ckpt/"+str(learning_rate)):
                    os.makedirs("ckpt/"+str(learning_rate))
                torch.save({"model":model.module.state_dict(),"optim":optimizer.state_dict()},\
                    "./ckpt/{}/train_{}_step.pth".format(learning_rate,1+step))
                gc.collect()
            total_loss.backward()
            optimizer.step()
        else:
            #val mode
            print("###one training epoch ends. Now validation###")
            epoch += 1
            valid_gaze = []
            valid_total = []
            valid_head = []
            valid_eye = []

            dataiterator = iter(valid_loader)
            while True:
                try:
                    input_data = next(dataiterator)
                except StopIteration:
                    break
                for key in input_data:
                    if key!="img_name":
                        if use_cuda:
                            input_data[key] = Variable(input_data[key]).type(torch.cuda.FloatTensor)
                        else:
                            input_data[key] = Variable(input_data[key]).type(torch.FloatTensor)
                img_head = input_data['head_image']
                img_leye = input_data['leye_image']
                img_reye = input_data['reye_image']
                head_gt = input_data['head_lola']
                gaze_gt = input_data['gaze_lola']
                eye_gt = input_data['eye_lola']
                head_gt = (head_gt + 90)/180
                gaze_gt = (gaze_gt + 90)/180
                eye_gt = (eye_gt + 90)/180
                output = model(img_head,img_leye,img_reye)
                
                loss_fn = SmoothL1Loss()
                loss_head = loss_fn(output['head'],head_gt).item()
                loss_eye = loss_fn(output['eye'],eye_gt).item()
                loss_gaze = loss_fn(output['gaze'],gaze_gt).item()
                valid_gaze.append(loss_gaze)
                valid_head.append(loss_head)
                valid_eye.append(loss_eye)
                valid_total.append(loss_head + loss_eye + loss_gaze)
            print("head: {:.5f} eye: {:.5f} gaze: {:.5f} total: {:.5f}"\
                .format(np.mean(valid_head),np.mean(valid_eye),np.mean(valid_gaze),np.mean(valid_total)))
            print("epoch {}#########################################".format(epoch))
            phase = "train"
            train_loader,valid_loader = get_train_valid_loader(gaze_set,8*GPU_COUNT)
            dataiterator = iter(train_loader)
            if not os.path.exists("ckpt/"+str(learning_rate)):
                os.makedirs("ckpt/"+str(learning_rate))
            torch.save({"model":model.module.state_dict(),"optim":optimizer.state_dict()},\
                "./ckpt/{}/{}_epoch.pth".format(learning_rate,epoch))
            gc.collect()



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
        img_eye = input_data['eye_image']
        output = model(img_head,img_eye)
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

    transform = iaa.Sequential([
    iaa.CropAndPad(percent=(-0.1, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
    iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255)),
        ], random_order=True) # apply augmenters in random order


    gaze_set = GazeDataset('/data/mc_data/MC4/train',"train",transform)
    #pretrained_model = "./ckpt/0.005/11_epoch.pth"
    pretrained_model=None
    main(gaze_set,pretrained_model)
    test_data_dir = "/data/mc_data/MC4/test"
    output_path = "./pred.txt"
    model_path = "./ckpt/0.005/train_1000_step.pth"
    test_loader = get_test_loader(test_data_dir)
    output_predict(test_loader,output_path,model_path)
