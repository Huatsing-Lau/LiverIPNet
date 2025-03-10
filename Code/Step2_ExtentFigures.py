# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import argparse
from collections import Counter
import copy
from glob import glob

import json
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np

import os
import pandas as pd
import pdb
import random

import scipy
import SimpleITK as sitk

from sklearn import metrics
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import DataLoader, sampler, Dataset
import torch.nn as nn
import torch.nn.functional as F
from models.model_resnet import Resmode
from utils import *

# + code_folding=[0]
# Training settings
parser = argparse.ArgumentParser(description='Configurations for Mission1 Training')

parser.add_argument('--max_epoch',
                    type=int,
                    default=100,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr',
                    type=float,
                    default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--fold',
                    type=int,
                    default=10,
                    help='number of folds (default: 1)')
parser.add_argument('--results_dir',
                    default='./results',
                    help='results directory (default: ./results)')

parser.add_argument('--log_data',
                    action='store_true',
                    default=False,
                    help='log data using tensorboard')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out',
                    action='store_true',
                    default=False,
                    help='enabel dropout (p=0.25)')
parser.add_argument('--backbone_requires_grad',
                    action='store_true',
                    default=True,
                    help='whether to train the backbone')
parser.add_argument('--input_channel',
                    type=int,
                    default=5,
                    help='number of input image channels')
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
args = parser.parse_args(args=[])  # args=[]
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.n_classes = 2
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


# + code_folding=[0]
# Dataset
class M1_dataset(Dataset):
    def __init__(self, path_list, label_list, mode, aug=0.3):
        self.All_data = path_list
        self.All_label = label_list
        self.mode = mode
        self.aug = aug
        self.phase = ['_T1', '_T2', '_AP', '_PP', '_20 MIN']

        self.index_0 = [x for x, y in list(enumerate(label_list)) if y == 0]
        self.index_1 = [x for x, y in list(enumerate(label_list)) if y == 1]
        

    def __getitem__(self, index):
           # transformer
        def random_drop(img_clip, max_rate=0.2):
            c = img_clip.shape[1]
            d = img_clip.shape[2]
            # 随机生成一个立方体
            cut_c = np.random.rand(1)
            cut_d = np.random.rand(1)

            while cut_c*cut_d > max_rate:
                if np.random.rand(1) > 0.5:
                    cut_c = max(cut_c*0.8, 0.01)
                else:
                    cut_d = max(cut_d*0.8, 0.01)

            drop_d1 = np.random.randint(0, d-d*cut_d+1)
            drop_c1 = np.random.randint(0, c-c*cut_c+1)

#             if np.random.rand(1)>0.5:
            img_clip[:,int(drop_c1):int(min(drop_c1+c*cut_c, c)),
                     int(drop_d1):int(min(drop_d1+d*cut_d, d))] = np.zeros(( 25,
                                                                           (int(min(drop_c1+c*cut_c, c)-drop_c1)),
                                                                           (int(min(drop_d1+d*cut_d, d)-drop_d1))))
#             else:
#                 img_clip[:,int(drop_c1):int(min(drop_c1+c*cut_c, c)),
#                          int(drop_d1):int(min(drop_d1+d*cut_d, d))] = np.ones(( 5,
#                                                                                (int(min(drop_c1+c*cut_c, c)-drop_c1)),
#                                                                                (int(min(drop_d1+d*cut_d, d)-drop_d1))))
            return img_clip

        def random_clip(img_clip, max_rate=0.1):
            a = img_clip.shape[0]
            c = img_clip.shape[2]
            d = img_clip.shape[3]
            drop_rate1 = np.random.randint(0, max_rate*100) / 100
            drop_rate2 = np.random.randint(0, max_rate*100) / 100

            if (np.random.rand(1) > 0.5) and (a > 10):
                img_clip = img_clip[int(a * drop_rate1):
                                    int(a * (1 - drop_rate2)), :, :, :]
            if (np.random.rand(1) > 0.5) and (c > 10):
                img_clip = img_clip[:, :, int(c * drop_rate1):
                                    int(c * (1 - drop_rate2)), :]
            if (np.random.rand(1) > 0.5) and (d > 10):
                img_clip = img_clip[:, :, :, int(d * drop_rate1):
                                    int(d * (1 - drop_rate2))]
            return img_clip

        def random_flip(img_clip, flip_rate=0.5):
            if np.random.rand(1) > flip_rate:
                img_clip = img_clip[:, ::-1, :].copy()
            if np.random.rand(1) > flip_rate:
                img_clip = img_clip[:, :, ::-1].copy()
            return img_clip

        def resize_np(img, img_new):
            a = img.shape[0]
            c = img.shape[1]
            a_new = img_new.shape[0]
            c_new = img_new.shape[1]
            img = scipy.ndimage.zoom(
                img, (a_new/a,  c_new/c))
            return img

        def copy_channel(img, channel=0):
            img_clip = np.concatenate(
                [img_clip.copy(), img_clip.copy()], axis=channel)

        def mix_up(img, target):
            if target == 0:
                index_supply = random.choice(self.index_0)
            else:
                index_supply = random.choice(self.index_1)
            name_file_supply = self.All_data[index_supply]

            # 读取新的数据
            supply_img = np.zeros((25, 32, 32))
            n = 0

            for phase_id in range(5):
                channel_name = name_file_supply[:-4] + self.phase[phase_id] + '.npy'
                img_channel = np.load(channel_name)

                img1 = np.max(img_channel, axis=0)
                img2 = np.mean(img_channel, axis=0)
                img3 = np.min(img_channel, axis=0)
                img4 = img1-img3
                img5 = np.median(img_channel, axis=0)

                for img_fill in [img1, img2, img3, img4, img5]:
                    img_fill = img_fill/(img1.max()+0.01)
                    img_fill = resize_np(img_fill, np.zeros((32, 32)))
                    supply_img[n, :, :] = img_fill
                    n+=1
            
            ratio = np.random.rand(1)/2+0.5
            img = img*ratio + supply_img*(1-ratio)
            return img


        name_file, target = self.All_data[index], self.All_label[index]
        output_img= np.zeros((25, 32, 32))
        n = 0
        
        for phase_id in range(5):
            channel_name = name_file[:-4] + self.phase[phase_id] + '.npy'
            img_channel = np.load(channel_name)
            
            img1 = np.max(img_channel, axis=0)
            img2 = np.mean(img_channel, axis=0)
            img3 = np.min(img_channel, axis=0)
            img4 = img1-img3
            img5 = np.median(img_channel, axis=0)

            for img_fill in [img1, img2, img3, img4, img5]:
                img_fill = img_fill/(img1.max()+0.01)
                img_fill = resize_np(img_fill, np.zeros((32, 32)))
                output_img[n, :, :] = img_fill
                n+=1

        if self.mode =='train':
            # 随机翻转
            if np.random.rand(1)<1/3:
                output_img = mix_up(output_img, target=target)
            elif np.random.rand(1)<1/2:
                output_img = random_drop(output_img, max_rate=self.aug)
            output_img = random_flip(output_img, flip_rate=0.5)


        return output_img, target, name_file[:-4]

    def __len__(self):
        return len(self.All_data)

# +
## 读取数据与模型
filepath_df = pd.read_csv('../All_keep_v3/label_v4.csv')
filepath_list = filepath_df.path
label_list = filepath_df.label

# dataset object
my_dataset = M1_dataset(path_list=list(filepath_list),
                           label_list=list(label_list), mode='test', aug=0)
# dataloader
my_loader = DataLoader(my_dataset, batch_size=16, shuffle=False)

# model initialize
model = Resmode(input_channel=25).to(args.device)
model.load_state_dict(
    torch.load('../log3/modelv5_fold0_auc0.943_balance_1123.pth'))
# -

# # 打印所有id

result_in = pd.read_csv('result_in.csv')
name_ids = []
for i in range(len(result_in)):
    if result_in.label[i] == 0 and result_in.pred[i] == 1:
        p = result_in.path[i]
        p = p.split('/')[-1]
        name_ids.append(p.split('.')[0])
name_ids


# # 病灶实例展示

def resize_np(img, img_new):
    a = img.shape[0]
    c = img.shape[1]
    a_new = img_new.shape[0]
    c_new = img_new.shape[1]
    img = scipy.ndimage.zoom(
        img, (a_new/a,  c_new/c))
    return img


# + code_folding=[3]
name_ids = ['664488_HCC_01']
cmap='gist_gray'

for name_id in name_ids:
#     name_file = '../All_keep_v3/' + name_id
#     name_file = '../Add_sample/Revise V3/' + name_id
    name_file = '../All_outsource_v5/ZSYY/' + name_id

    phase = ['_T1', '_T2', '_AP', '_PP', '_20 MIN']

    img_T1 = np.load(name_file + '_T1.npy')
    img_T2 = np.load(name_file + '_T2.npy')
    img_AP = np.load(name_file + '_AP.npy')
    img_PP = np.load(name_file + '_PP.npy')
    img_20 = np.load(name_file + '_20 MIN.npy')
    img_origin=[img_T1, img_T2, img_AP, img_PP, img_20]
    output_img= np.zeros((25, 32, 32))

    plt.subplots(figsize=[15,15])
    n = 1
    k=0
    for i in range(5):
        temp = img_origin[i]
        temp1 = temp[int(temp.shape[0]/2), :, :]
        temp1 = resize_np(temp1, np.zeros((32, 32)))
        plt.subplot(5,5,n)
        plt.imshow(temp1, cmap=cmap)
        if i == 0:
            plt.ylabel('T1')
        elif i == 1:
            plt.ylabel('T2')
        elif i == 2:
            plt.ylabel('AP')
        elif i == 3:
            plt.ylabel('PVP')
        else:
            plt.ylabel('HBP')
        plt.xticks([])
        plt.yticks([])

        n += 1

        img_channel = temp
        img1 = np.max(img_channel, axis=0)
        img2 = np.mean(img_channel, axis=0)
        img3 = np.min(img_channel, axis=0)
        img4 = img1-img3
        img5 = np.median(img_channel, axis=0)

        plt.subplot(5,5,n)
        plt.imshow(resize_np(img1, np.zeros((32, 32))), cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        n += 1

        plt.subplot(5,5,n)
        plt.imshow(resize_np(img2, np.zeros((32, 32))), cmap=cmap)   
        plt.xticks([])
        plt.yticks([])
        n += 1

        plt.subplot(5,5,n)
        plt.imshow(resize_np(img3, np.zeros((32, 32))), cmap=cmap)   
        plt.xticks([])
        plt.yticks([])
        n += 1

        plt.subplot(5,5,n)
        plt.imshow(resize_np(img4, np.zeros((32, 32))), cmap=cmap)   
        plt.xticks([])
        plt.yticks([])
        n += 1    
                    
        for img_fill in [img1, img2, img3, img4, img5]:
            img_fill = img_fill/(img1.max()+0.01)
            img_fill = resize_np(img_fill, np.zeros((32, 32)))
            output_img[k, :, :] = img_fill
            k+=1

#     temp = img_origin[0]
#     temp1 = temp[int(temp.shape[0]/2), :, :]
#     img_channel = temp
#     img11 = np.max(img_channel, axis=0)
#     img21 = np.mean(img_channel, axis=0)
#     img31 = np.min(img_channel, axis=0)
#     img41 = img11-img31

#     temp = img_origin[2]
#     temp2 = temp[int(temp.shape[0]/2), :, :]
#     img_channel = temp
#     img13 = np.max(img_channel, axis=0)
#     img23 = np.mean(img_channel, axis=0)
#     img33 = np.min(img_channel, axis=0)
#     img43 = img13-img33

#     plt.subplot(6,5,26)
#     plt.imshow(resize_np(temp2, np.zeros((32, 32))) - resize_np(temp1, np.zeros((32, 32))))
#     plt.xlabel('middle layer')
#     plt.ylabel('AP - T1')

#     plt.subplot(6,5,27)
#     plt.imshow(resize_np(img13, np.zeros((32, 32))) - resize_np(img11, np.zeros((32, 32))))
#     plt.xlabel('maximum of layers')

#     plt.subplot(6,5,28)
#     plt.imshow(resize_np(img23, np.zeros((32, 32))) - resize_np(img21, np.zeros((32, 32))))  
#     plt.xlabel('mean of layers')

#     plt.subplot(6,5,29)
#     plt.imshow(resize_np(img33, np.zeros((32, 32))) - resize_np(img31, np.zeros((32, 32))))
#     plt.xlabel('minimum of layers')

#     plt.subplot(6,5,30)
#     plt.imshow(resize_np(img43, np.zeros((32, 32))) - resize_np(img41, np.zeros((32, 32))))
#     plt.xlabel('maximum - minimum')
    
#     plt.savefig('figures/'+name_id + '.png')


# + code_folding=[0]
# 尝试不同配色
plt.subplots(dpi=200)

plt.subplot(1,5,1)
plt.imshow(img1)
plt.xticks([])
plt.yticks([])

plt.subplot(1,5,2)
img1_ = (img1-img1.min())/(img1.max()-img1.min())
plt.imshow(img1_, cmap='spring')
plt.xticks([])
plt.yticks([])

plt.subplot(1,5,3)
plt.imshow(img1, cmap='cividis')
plt.xticks([])
plt.yticks([])

plt.subplot(1,5,4)
plt.imshow(img2, cmap='gray')
plt.xticks([])
plt.yticks([])

plt.subplot(1,5,5)
plt.imshow(1-img1/2529, cmap='gray')
plt.xticks([])
plt.yticks([])
# -

# # 模型可视化展示

# +
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

plt.subplots(figsize=(9, 9))
n=1
# gradcam目标层通常是最后一个卷积层
for i in range(3):
    target_layers = [model.resnet_baseline.layer1[i].conv1]
    # model.resnet_baseline.layer3[5].conv3
    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers,
                  use_cuda=True if device == 'cuda' else False)  # , uses_gradients=False

    input_tensor = torch.tensor(np.expand_dims(
        output_img, 0)).cuda().to(torch.float32)
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    # grayscale_cam = cam(input_tensor=input_tensor, targets=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)

    plt.subplot(3, 3, n)
    plt.imshow(grayscale_cam[0])
    # visualization = show_cam_on_image(img1, grayscale_cam)  # (224, 224, 3)
    n+=1
    
    target_layers = [model.resnet_baseline.layer1[i].conv2]
    cam = GradCAM(model=model, target_layers=target_layers,
              use_cuda=True if device == 'cuda' else False)  # , uses_gradients=False
    input_tensor = torch.tensor(np.expand_dims(
        output_img, 0)).cuda().to(torch.float32)
    grayscale_cam = cam(input_tensor=input_tensor)
    plt.subplot(3, 3, n)
    plt.imshow(grayscale_cam[0])
    n+=1
    
    target_layers = [model.resnet_baseline.layer1[i].conv3]
    cam = GradCAM(model=model, target_layers=target_layers,
              use_cuda=True if device == 'cuda' else False)  # , uses_gradients=False
    input_tensor = torch.tensor(np.expand_dims(
        output_img, 0)).cuda().to(torch.float32)
    grayscale_cam = cam(input_tensor=input_tensor)
    plt.subplot(3, 3, n)
    plt.imshow(grayscale_cam[0])
    n+=1
# -

# 不同配色展示
target_layers = [model.resnet_baseline.layer1[2].conv3]
cam = GradCAM(model=model, target_layers=target_layers,
          use_cuda=True if device == 'cuda' else False)  # , uses_gradients=False
input_tensor = torch.tensor(np.expand_dims(
    output_img, 0)).cuda().to(torch.float32)
grayscale_cam = cam(input_tensor=input_tensor)
plt.imshow(1-grayscale_cam[0], cmap='jet')
plt.xticks([])
plt.yticks([])
plt.colorbar()

# + code_folding=[8]
# 另外一种版本
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
 
def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
 
    # 获取模型输出的feature/score
    model.eval()
    features = model.features(img)
    output = model.classifier(features)
 
    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g
 
    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]
 
    features.register_hook(extract)
    pred_class.backward() # 计算梯度
 
    grads = features_grad   # 获取梯度
 
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
 
    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 512是最后一层feature的通道数
    for i in range(512):
        features[i, ...] *= pooled_grads[i, ...]
 
    # 以下部分同Keras版实现
    heatmap = features.detach().numpy()
    heatmap = np.mean(heatmap, axis=0)
 
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
 
    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()
 
    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘
 


# -

# # 模型参数

# +
## 计算单次推理的时间
import time
T1 = time.time()

(inputs, label, _) = test_loader.dataset.__getitem__(0)
inputs = torch.tensor(inputs).to(torch.float32).cuda().unsqueeze(0)
outputs = model(inputs)
T2 = time.time()
print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))

# +
## 计算模型总参数量
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total))

# 查看模型每层
model.state_dict

# +
## 画出训练的loss过程 
# Load
history = np.load('trainig_history.npy').item()

plt.subplots(figsize=(9,4),dpi=300)
plt.subplot(1,2,1)
plt.plot(np.array(history['train_acc'])/100, label='Train')
plt.plot(np.array(history['test_acc'])/100, 'orange', label='Test')
plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Model accuracy')

plt.subplot(1,2,2)
plt.plot(np.array(history['train_loss'])*16, label='Train')
plt.plot(np.array(history['test_loss'])*16, 'orange', label='Test')
plt.legend()
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Model loss')
plt.rc('font',family='Times') 
matplotlib.rcParams.update({'font.size': 12}) 

