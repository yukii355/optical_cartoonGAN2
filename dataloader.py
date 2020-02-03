import torch
import torch.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os, random
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage, Resize, RandomCrop


# This is a dataloader for new opticalflow GAN cartoonize experiment.

# I referenced ReCoNet dataset.py code.

# This dataloader's purpose is try to load dataset(img2 and opticalflow folder)


# Step1. import img2 dataset
# Step2. import opticalflow
# Step3. transform the dataset (if it is needed)
# Step4. taking Dataset as a batch




'''
# import img2 datafolder
img2_dataset = dset.ImageFolder(root='/home/moriyama/PycharmProjects/op_background/img2/')
img2_dataloader = DataLoader(img2_dataset, batch_size=2, shuffle=False, num_workers=None)
'''




'''
path = "/home/moriyama/Downloads/Melibea-Wakeup.mp4"
def video_loader(path = path, type=):
    data_list = []
video_capture = cv2.VideoCapture("/home/moriyama/Downloads/Boat1193.mp4")
output_file = "./output.mp4"
'''

def cal_flow(prvs, next):
    prvs = cv2.cvtColor(prvs,cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(next,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
def transform():


    return Compose([  ToPILImage(),  ToTensor(),
                     Normalize(means, stds),
    ])

class Img2_Dataset(Dataset):

    def __init__(self, img_path, ani_path, transforms=None):

        """
        Initializes internal Module state, shared by both nn.Module and
        ScriptModule.
        """

        self.img_path = img_path
        self.ani_path = ani_path
        self.transform = transforms


    def __len__(self):

        return len(os.listdir(self.img_path))


    def __getitem__(self, idx):
        randomflip = random.randint(0, 1)
        randomcrop_x = random.randint(0, 32)
        randomcrop_y = random.randint(0, 32)

        if idx == 2760:
            num1 = "%06d.jpg" %(idx + 1)

            if randomflip:

                img1 = cv2.rotate(cv2.resize(cv2.imread(self.img_path + num1), (256,256)), cv2.ROTATE_180)[randomcrop_x:randomcrop_x+224, randomcrop_y:randomcrop_y+224]
                img2 = cv2.rotate(cv2.resize(cv2.imread(self.img_path + num1), (256,256)), cv2.ROTATE_180)[randomcrop_x:randomcrop_x+224, randomcrop_y:randomcrop_y+224]


                flow = np.transpose(cal_flow(img1,img2), [2, 0, 1])
            else:
                img1 = cv2.resize(cv2.imread(self.img_path + num1), (256,256))[randomcrop_x:randomcrop_x+224, randomcrop_y:randomcrop_y+224]
                img2 = cv2.resize(cv2.imread(self.img_path + num1), (256,256))[randomcrop_x:randomcrop_x+224, randomcrop_y:randomcrop_y+224]

                flow = np.transpose(cal_flow(img1, img2), [2, 0, 1])
        else:
            num1 = "%06d.jpg" %(idx + 1)
            num2 = "%06d.jpg" %(idx + 2)
            if randomflip:
                img1 = cv2.rotate(cv2.resize(cv2.imread(self.img_path + num1), (256,256)),cv2.ROTATE_180)[randomcrop_x:randomcrop_x+224, randomcrop_y:randomcrop_y+224]
                img2 = cv2.rotate(cv2.resize(cv2.imread(self.img_path + num2), (256,256)), cv2.ROTATE_180)[randomcrop_x:randomcrop_x+224, randomcrop_y:randomcrop_y+224]

            # op = os.path.join(self.op_path, str(idx + 1) + ".pkl")
            # with open(op, "rb") as f:
            #     flow = pickle.load(f)
                flow = np.transpose(cal_flow(img1,img2), [2, 0, 1])
            else:
                img1 = cv2.resize(cv2.imread(self.img_path + num1), (256,256))[randomcrop_x:randomcrop_x+224, randomcrop_y:randomcrop_y+224]
                img2 = cv2.resize(cv2.imread(self.img_path + num2), (256,256))[randomcrop_x:randomcrop_x+224, randomcrop_y:randomcrop_y+224]

                # op = os.path.join(self.op_path, str(idx + 1) + ".pkl")
                # with open(op, "rb") as f:
                #     flow = pickle.load(f)
                flow = np.transpose(cal_flow(img1, img2), [2, 0, 1])
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        flow = torch.from_numpy(flow)
        ani_file = os.listdir(self.ani_path)
        rand = random.randint(0,2953)
        ani_path = os.path.join(self.ani_path, ani_file[rand])
        ani = cv2.resize(cv2.imread(ani_path), (256,256))[randomcrop_x:randomcrop_x+224, randomcrop_y:randomcrop_y+224]
        ani = cv2.cvtColor(ani, cv2.COLOR_BGR2RGB)
        ani = self.transform(ani)


        return img1, img2, ani, flow


if __name__ == '__main__':
    img_path = "/home/moriyama/PycharmProjects/op_background/img5/"
    ani_path = "/home/moriyama/PycharmProjects/animation_popeye_fixed/"
    dataset = Img2_Dataset(img_path,ani_path, transform())

    loader = DataLoader(dataset, batch_size=8,num_workers=8)
    for e in range(5):
        for i,data in enumerate(loader):
            img1, img2, ani, op = data
            print(img1.size(),e)
        # print(img2.size())
        # print(op.size())
        # #
        # print(i)