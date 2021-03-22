import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import imageio
import natsort
import os
import os.path
import multiprocessing
from PIL import Image
import matplotlib
import scipy.misc
from torchvision import transforms
from torch.autograd import Variable
from utils.myTransforms import denorm, norms, detransformcv2
from utils.myDatasets import ImagePair, Im_transform
from rebuild_313_train import myNet
import cv2

def im2double(im):
    double_im = cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return double_im


def latent_lrr(X):
    A = X
    x_lambda = 0.8
    tol = 1e-6
    rho = 1.1
    max_mu = 1e6
    mu = 1e-6
    maxIter = 1e6
    d, n = X.shape
    m = A.shape[1]
    XZZ = X.T
    atx = np.dot(XZZ, X)
    inv_a = np.linalg.inv(np.dot(XZZ, A) + np.eye(m))
    inv_b = np.linalg.inv(np.dot(A, XZZ) + np.eye(d))

    #  Initializing optimization variables
    J = np.zeros((m, n))
    Z = np.zeros((m, n))
    L = np.zeros((d, d))
    S = np.zeros((d, d))

    E = np.zeros((d, n))

    Y1 = np.zeros((d, n))
    Y2 = np.zeros((m, n))
    Y3 = np.zeros((d, d))

    # %% Start main loop
    iter = 0
    while iter < maxIter:
        iter = iter + 1
        temp_J = Z + Y2 / mu
        [U_J, sigma_J, V_J] = np.linalg.svd(temp_J)

        if sigma_J.ndim == 1:
            sigma_J = sigma_J
        else:
            sigma_J = np.diag(sigma_J)

        svp_J = np.sum(sigma_J > 1 / mu)
        if svp_J >= 1:
            sigma_J = sigma_J[0:svp_J] - 1 / mu
        else:
            svp_J = 1
            sigma_J = [0]

        J = np.dot(np.dot(U_J[:, 0: svp_J], np.diag(sigma_J)), (V_J[0: svp_J, :]))

        # updating S by the Singular Value Thresholding(SVT) operator
        temp_S = L + Y3 / mu

        [U_S, sigma_S, V_S] = np.linalg.svd(temp_S)
        if sigma_S.ndim == 1:
            sigma_S = sigma_S
        else:
            sigma_S = np.diag(sigma_S)

        svp_S = np.sum(sigma_S > 1 / mu)
        if svp_S >= 1:
            sigma_S = sigma_S[0:svp_S] - 1 / mu
        else:
            svp_S = 1
            sigma_S = [0]

        S = np.dot(np.dot(U_S[:, 0: svp_S], np.diag(sigma_S)), (V_S[0: svp_S, :]))

        # udpate Z L E
        Z = np.dot(inv_a, (atx - np.dot(np.dot(XZZ, L), X) - np.dot(XZZ, E) + J + ((np.dot(XZZ, Y1) - Y2) / mu)))
        L = np.dot(np.dot((X - np.dot(X, Z) - E), XZZ) + S + (np.dot(Y1, XZZ) - Y3) / mu, inv_b)

        xmaz = X - np.dot(X, Z) - np.dot(L, X)
        temp = xmaz + Y1 / mu
        E1 = np.maximum(0, temp - x_lambda / mu)
        E2 = np.minimum(0, temp + x_lambda / mu)
        E = E1 + E2

        leq1 = xmaz - E
        leq2 = Z - J
        leq3 = L - S
        max_l1 = np.max(np.max(np.abs(leq1)))
        max_l2 = np.max(np.max(np.abs(leq2)))
        max_l3 = np.max(np.max(np.abs(leq3)))
        stopC1 = max(max_l1, max_l2)
        stopC = max(stopC1, max_l3)

        if stopC < tol:
            # print('when iter= ', iter, '----------LRR done.')
            break
        else:
            Y1 = Y1 + np.dot(mu, leq1)
            Y2 = Y2 + np.dot(mu, leq2)
            Y3 = Y3 + np.dot(mu, leq3)
            mu = min(max_mu, np.dot(mu, rho))

    return Z, L, E


def lrr_process(im_path):
    # print(im_path)
    X0 = cv2.imread(im_path)
    X1 = cv2.cvtColor(X0, cv2.COLOR_BGR2GRAY)
    X = im2double(X1)
    Z, L, E = latent_lrr(X)

    lrr = np.dot(X, Z)
    lrr = np.maximum(lrr, 0)
    lrr = np.minimum(lrr, 1)

    lrr = Image.fromarray(np.uint8(lrr * 255))

    saliency = np.dot(L, X)
    saliency = np.maximum(saliency, 0)
    saliency = np.minimum(saliency, 1)

    saliency = Image.fromarray(np.uint8(saliency * 255))

    return lrr, saliency


def tolrrandtransform(pathct,pathmr):
    # LRR segmentation
    lrr1, saliency1 = lrr_process(pathct)
    lrr2, saliency2 = lrr_process(pathmr)
    pair_loader0 = ImagePair(impath1=pathct, impath2=pathmr,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=mean, std=std)
                             ]))
    imgct, imgmr = pair_loader0.get_pair()

    pair_1 = Im_transform(im1=lrr1, im2=saliency1,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=mean, std=std)
                          ]))
    img11, img12 = pair_1.get_pair()  # LRR for ct
    pair_2 = Im_transform(im1=lrr2, im2=saliency2,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=mean, std=std)
                          ]))
    img21, img22 = pair_2.get_pair()  # LRR for mr

    imgct.unsqueeze_(0)
    imgmr.unsqueeze_(0)
    imgct = Variable(imgct.cpu(), requires_grad=False)
    imgmr = Variable(imgmr.cpu(), requires_grad=False)

    img11.unsqueeze_(0)
    img12.unsqueeze_(0)
    img11 = Variable(img11.cpu(), requires_grad=False)
    img12 = Variable(img12.cpu(), requires_grad=False)

    img21.unsqueeze_(0)
    img22.unsqueeze_(0)
    img21 = Variable(img21.cpu(), requires_grad=False)
    img22 = Variable(img22.cpu(), requires_grad=False)
    return imgct, imgmr,img11, img12,img21, img22

if __name__=="__main__":
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    image_length = 256
    image_width = 256
    mean = [0, 0, 0]
    std = [1, 1, 1]
    # load the model
    model = myNet()
    model.load_state_dict(torch.load('./DB_DFFuse.pth', map_location='cpu'))
    model.eval()

    root_path1 = './MRandPET(Normal)/MR/'
    root_path2 = './MRandPET(Normal)/PET/'
    file1=[]
    file2 = []
    for root,dir,files1 in os.walk(root_path1):
        for file in files1:
            pathct=root_path1+str(file)
            file1.append(pathct)
    for root, dir, files2 in os.walk(root_path2):
        for file in files2:
            pathmr = root_path2 + str(file)
            file2.append(pathmr)

    for i in range(len(file1)):
        imgct, imgmr, img11, img12, img21, img22 = tolrrandtransform(file1[i], file2[i])
        name2=str(i)

        f1_sb = model.spatial_work(img11, img12)
        f1_cb = model.channel_to_one(img11, img12)
        f1 = model.fusion2(f1_sb, f1_cb)
        f2_sb = model.spatial_work(img21, img22)
        f2_cb = model.channel_to_one(img21, img22)
        f2 = model.fusion2(f2_sb, f2_cb)

        res1 = model.tensor_mean([f1, imgct])
        res2 = model.tensor_mean([f2, imgmr])
        res12 = model.tensor_max([res1, res2])

        res = denorm(mean, std, res12[0]).clamp(0, 1) * 255
        res_img = res.cpu().data.numpy().astype('uint8')
        img = res_img.transpose([1, 2, 0])

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        imageio.imsave('./Results(mr_pet)/io_' + name2 +'.png', img)
        matplotlib.image.imsave('./Results(mr_pet)/mb_' + name2 + '.png', img)




