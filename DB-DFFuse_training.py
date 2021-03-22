import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import time
import matplotlib.pyplot as plt
import numpy as np
import glob
import imageio
import natsort
from pytorch_msssim import ssim
import os
import os.path
import multiprocessing
import scipy.io as scio
from PIL import Image
import cv2
# device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
def denorm(mean=[0, 0, 0], std=[1, 1, 1], tensor=None):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def load_onemodal():
    # load the train mri data
    dataset = os.path.join(os.getcwd(), './data_add_t2pet/MR-T2/')
    data = glob.glob(os.path.join(dataset, "*.*"))
    data = natsort.natsorted(data, reverse=False)
    # print(len(data))
    train_one = np.zeros((len(data), image_width, image_length,3), dtype=float) #(272,256,256,3)
    data_1 = np.zeros((len(data), image_width, image_length, 3))
    for i in range(len(data)):
        Im = Image.open(data[i]) #(256,256)
        data_1[i, :, :,:] = Im.convert('RGB') #(256,256,3)
        # print(data[i].size)
        train_one[i, :, :,:] = (data_1[i, :, :,:] - np.min(data_1[i, :, :,:])) / (
                np.max(data_1[i, :, :,:]) - np.min(data_1[i, :, :,:]))
        train_one[i, :, :,:] = np.float32(train_one[i, :, :,:])
    # print(train_one.shape)
    train_one = train_one.transpose([0,3,1, 2])
    # print(train_one.shape)
    train_one_tensor = torch.from_numpy(train_one).float()
    print('load_onemodal ',train_one_tensor.shape)

    return train_one_tensor

def second_onemodal():
    # load the train pet data
    dataset = os.path.join(os.getcwd(), './data_add_t2pet/FDG/')
    data = glob.glob(os.path.join(dataset, "*.*"))
    data = natsort.natsorted(data, reverse=False)
    train_other = np.zeros((len(data), image_width, image_length, pet_channels), dtype=float)
    train_pet = np.zeros((len(data), image_width, image_length), dtype=float)
    train_one = np.zeros((len(data), image_width, image_length,3), dtype=float) #(272,256,256,3)
    data_2 = np.zeros((len(data), image_width, image_length, 3))
    for i in range(len(data)):
        # train_pet[i, :, :] = (imageio.imread(data[i]))
        train_other[i, :, :, :] = (imageio.imread(data[i]))
        train_pet[i, :, :] = 0.2989 * train_other[i, :, :, 0] + 0.5870 * train_other[i, :, :, 1] + 0.1140 * train_other[i, :, :, 2]

        Im = Image.fromarray(train_pet[i, :, :]) #numpy to Image
        data_2[i, :, :,:] = Im.convert('RGB')  # (256,256,3)
        train_one[i, :, :, :] = (data_2[i, :, :,:] - np.min(data_2[i, :, :,:])) / (
                np.max(data_2[i, :, :,:]) - np.min(data_2[i, :, :,:]))
        train_one[i, :, :, :] = np.float32(train_one[i, :, :,:])
    # print(train_one.shape)
    train_one = train_one.transpose([0, 3, 1, 2])
    # print(train_one.shape)
    train_one_tensor = torch.from_numpy(train_one).float()
    print('second_onemodal ',train_one_tensor.shape)

    return train_one_tensor

class ConvBlock(nn.Module):
    def __init__(self, inplane, outplane):
        super(ConvBlock, self).__init__()
        self.padding = (1, 1, 1, 1)
        self.conv = nn.Conv2d(inplane, outplane, kernel_size=3, padding=0, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)

        # self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        out = F.pad(x, self.padding, 'replicate')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class DB_DFFuse(nn.Module):
    def __init__(self, resnet):
        super(DB_DFFuse, self).__init__()
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        self.conv5 = ConvBlock(256, 64)
        self.conv7 = ConvBlock(256, 256)
        self.conv8 = ConvBlock(256, 128)
        self.conv9 = ConvBlock(128, 64)
        self.conv10 = ConvBlock(192, 128)
        self.conv6 = ConvBlock(384, 256)
        self.conv11 = nn.Conv2d(64, 3, kernel_size=1, padding=0, stride=1, bias=True)

        # Initialize other parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        # Initialize conv1 with the pretrained resnet101
        for p in resnet.parameters():
            p.requires_grad = False
        self.conv1 = resnet.conv1
        self.conv1.stride = 1
        self.conv1.padding = (0, 0)

        self.pool = nn.MaxPool2d(2, 2)

    def tensor_max(self, tensors):
        max_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                max_tensor = tensor
            else:
                max_tensor = torch.max(max_tensor, tensor)
        return max_tensor

    def tensor_mean(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        mean_tensor = sum_tensor / len(tensors)
        return mean_tensor

    def tensor_sum(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        return sum_tensor

    def operate(self, operator, tensors):
        out_tensors = operator(tensors)
        return out_tensors

    def tensor_padding(self, tensors, padding=(1, 1, 1, 1), mode='constant', value=0):
        out_tensors = F.pad(tensors, padding, mode=mode, value=value)

        return out_tensors

    def spatial_work(self, tensor1, tensor2):
        # Feature extraction
        tensor1 = self.tensor_padding(tensors=tensor1, padding=(3, 3, 3, 3),
                                      mode='replicate')
        tensor2 = self.tensor_padding(tensors=tensor2, padding=(3, 3, 3, 3), mode='replicate')

        tensor1 = self.operate(self.conv1, tensor1)
        tensor2 = self.operate(self.conv1, tensor2)
        shape = tensor2.size()
        # spatial1 = tensor1.sum(dim=1, keepdim=True)
        spatial1 = tensor1.mean(dim=1, keepdim=True)
        spatial2 = tensor2.mean(dim=1, keepdim=True)
        # get weight map, soft-max
        EPSILON = 1e-5
        spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
        spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
        spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
        spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
        tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

        return tensor_f

    def spatial_work_one(self, tensor1):
        # Feature extraction
        tensor1 = self.tensor_padding(tensors=tensor1, padding=(3, 3, 3, 3),
                                      mode='replicate')
        tensor1 = self.operate(self.conv1, tensor1)
        shape = tensor1.size()
        # spatial1 = tensor1.sum(dim=1, keepdim=True)
        spatial1 = tensor1.mean(dim=1, keepdim=True)
        # get weight map, soft-max
        EPSILON = 1e-5
        spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + EPSILON)
        spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
        tensor_f = spatial_w1 * tensor1
        return tensor_f

    def channel_work(self, tensors):
        outs = self.tensor_padding(tensors=tensors, padding=(3, 3, 3, 3),
                                   mode='replicate')
        x1c1 = self.conv1(outs)
        x1p1=self.pool(x1c1)
        x1c2 = self.conv2(x1p1)
        x1c3 = self.conv3(x1c2)
        self.up = nn.Upsample(scale_factor=2)
        x1up = self.up(x1c3)

        x1pup = self.up(x1p1)
        x1c2up = self.up(x1c2)
        cat1 = torch.cat((x1up,x1pup, x1c2up), 1)
        x1c7 = self.conv7(cat1)

        x1c12 = self.conv2(x1c1)
        x1c123 = self.conv3(x1c12)
        x1c1234 = self.conv4(x1c123)

        out1 = self.tensor_max([x1c7, x1c1234])
        outx1 = self.conv5(out1)

        x1c8 = self.conv8(x1c7)
        x1c9 = self.conv9(x1c8)
        cat2 = torch.cat((x1c9, outx1), 1)
        x1catc9 = self.conv9(cat2)

        catx1 = torch.cat((x1catc9, outx1), 1)
        x1cat = self.conv9(catx1)

        ######
        x2c2 = self.conv2(x1c1) #1,64,256,256

        cat12 = torch.cat((x1c1, x2c2), 1)
        x2catc9=self.conv9(cat12) #128->64
        x2catc3=self.conv3(x2catc9)

        cat23 = torch.cat((x2c2, x2catc3), 1)
        x2catc10=self.conv10(cat23) #192->128
        x2catc4=self.conv4(x2catc10)

        cat34 = torch.cat((x2catc3, x2catc4), 1)
        x2catc11=self.conv6(cat34)
        x2cat=self.conv5(x2catc11)

        ##final connection
        cat = torch.cat((x2cat, x1cat), 1)
        xcatc9 = self.conv9(cat)
        out = self.conv2(xcatc9)
        return out

    def fusion2(self, *tensors):
        out = self.tensor_max(tensors)
        out = self.conv11(out)
        return out

    def fusion1(self, *tensors):
        out = self.tensor_max(tensors)
        # Feature reconstruction
        out = self.conv2(out)
        return out

    def channel_to_one(self,tensor1,tensor2):
        cb_1 = self.channel_work(tensor1)
        cb_2 = self.channel_work(tensor2)
        tensor1 = self.tensor_padding(tensors=tensor1, padding=(3, 3, 3, 3),
                                      mode='replicate')
        tensor2 = self.tensor_padding(tensors=tensor2, padding=(3, 3, 3, 3),
                                      mode='replicate')
        tensor1c1 = self.conv1(tensor1)
        tensor2c1 = self.conv1(tensor2)
        cb_1 = self.tensor_max([cb_1, tensor1c1])
        cb_2 = self.tensor_max([cb_2, tensor2c1])

        res1 = self.fusion1(cb_1, cb_2)
        return res1

    def forward(self, tensor1, tensor2):
        sb=self.spatial_work(tensor1,tensor2)
        cb=self.channel_to_one(tensor1,tensor2)
        res12 = self.fusion2(cb, sb)

        return res12

def myNet():

    resnet = models.resnet101(pretrained=True)
    mynet =DB_DFFuse(resnet).to(device)
    mynet=mynet.float()
    return mynet

def train_for_newModel(net,first_im,second_im):
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # optimize all dtn parameters
    l2_loss = nn.MSELoss()
    # perform the training
    counter = 0
    lamda = 0.7
    gamma_ssim = 0.5
    gamma_l2 = 0.5
    loss_history = []
    for epoch in range(EPOCH):
        # run batch images
        batch_idxs = 555 // batch_size
        for idx in range(0, batch_idxs):
            b_x = first_im[idx * batch_size: (idx + 1) * batch_size, :, :, :].to(device)
            b_y = second_im[idx * batch_size: (idx + 1) * batch_size, :, :, :].to(device)

            counter += 1
            output = net(b_x, b_y)  # output
            ssim_loss_mri = 1 - ssim(output, b_x, data_range=1)
            ssim_loss_pet = 1 - ssim(output, b_y, data_range=1)
            l2_loss_mri = l2_loss(output, b_x)
            l2_loss_pet = l2_loss(output, b_y)
            ssim_total = gamma_ssim * ssim_loss_mri + (1 - gamma_ssim) * ssim_loss_pet
            l2_total = gamma_l2 * l2_loss_mri + (1 - gamma_l2) * l2_loss_pet
            loss_total = lamda * ssim_total + (1 - lamda) * l2_total
            optimizer.zero_grad()  # clear gradients for this training step
            loss_total.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            loss_history.append(loss_total.item())

            if counter % 20 == 0:
                print(
                    "Epoch: [%2d],step: [%2d], mri_ssim_loss: [%.8f], pet_ssim_loss: [%.8f],  total_ssim_loss: [%.8f], total_l2_loss: [%.8f], total_loss: [%.8f]"
                    % (epoch, counter, ssim_loss_mri, ssim_loss_pet, ssim_total, l2_total, loss_total))

        if (epoch == EPOCH - 1):
            # Save a checkpoint
            torch.save(net.state_dict(), './ourDB_DFFuse_0314.pth')

    return loss_history


if __name__=="__main__":
    device='cuda:1' if torch.cuda.is_available() else 'cpu'
    print(device)
    image_length = 256
    image_width = 256
    mr_channels = 1
    gray_channels = 1
    pet_channels = 4
    rgb_channels = 2
    batch_size = 2
    EPOCH = 81
    learning_rate = 0.002
    mean = [0, 0, 0]  # normalization parameters
    std = [1, 1, 1]

    first_one=load_onemodal()
    second_one = second_onemodal()
    ourmodel = myNet()# Our model (DB_DFFuse)
    loss_history=train_for_newModel(ourmodel,first_one,second_one)

    plt.plot(loss_history, label='loss for every epoch')
    fig = plt.gcf()
    fig.savefig('./loss_0314.png')
    plt.show()
    print('finish fusing')
