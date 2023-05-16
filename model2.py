#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:45:02 2019

@author: liang
"""

import torch
from torch import nn
import torch.nn.functional as F

class TinyNet(nn.Module):

  #卷积神经网络，此处应用二维卷积神经网络

  def __init__(self):
    super(TinyNet, self).__init__()
    self.features = nn.Sequential(
                                # conv1_block ->15x15x32
                                nn.Conv2d(8, 16, 3, bias=False), #不需要偏置参数
                                # in_channels = 8, out_channels = 16, kernel_size = 4
                                # 输入张量用于确定权重，期望的四维输入张量数，卷积核大小
                                nn.BatchNorm2d(16),
                                #特征数量为16个
                                #nn.BatchNorm2d(3)，输入通道数为3，正常RGB通道
                                nn.ReLU(inplace=True),
                                nn.Conv2d(16, 16, 3, bias=False),
                                nn.BatchNorm2d(16),
                                nn.ReLU(inplace=True),
                                nn.InstanceNorm2d(16, affine=True),
                                #正则化
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2, 2, padding=1),
                                # conv2_block ->6x6x32
                                nn.Conv2d(16, 32, 3, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(True),
                                nn.Conv2d(32, 32, 3, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(True),
                                nn.InstanceNorm2d(32, affine=True),
                                nn.ReLU(True),
                                nn.MaxPool2d(2, 2, padding=1),
                                # conv3_block ->1x1x32
                                nn.AvgPool2d(6, 1),
                              )
    self.classifier = nn.Sequential(
                                nn.Linear(32, 64),
                                nn.ReLU(True),
                                nn.Linear(64, 4))
    
    
  def forward(self, xs):
    bs = xs.size(0)
    xs = self.features(xs)
    xs = xs.view(bs, -1)
    xs = self.classifier(xs)
    return xs
  
if __name__ == '__main__':
  tinynet = TinyNet()
  print(tinynet)
  xs = torch.randn(size=[128, 8, 32, 32])
  out = tinynet(xs)
  print(out.size())