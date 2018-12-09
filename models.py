import logging

import torch
import torch.nn as nn
import torch.utils.data
import torchvision

log = logging.getLogger(__name__)

class Yolo(nn.Module):
    def __init__(self):
        super(Yolo, self).__init__()
        # See https://github.com/pjreddie/darknet/blob/master/cfg/yolov1.cfg
        self.layer1 = nn.Sequential(
                #nn.BatchNorm2d(3),
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=7//2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(192),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=1//2),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=1//2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=1//2),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1//2),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1//2),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=3//2),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1)
        )
        self.layer6 = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),
        )
        self.layer7 = nn.Sequential(
                nn.Linear(in_features=7*7*1024, out_features=4096),
                nn.LeakyReLU(0.1),
                nn.Linear(in_features=4096, out_features=7*7*30)
        )

    def forward(self, x):
       x = self.layer1(x)
       x = self.layer2(x)
       x = self.layer3(x)
       x = self.layer4(x)
       x = self.layer5(x)
       x = self.layer6(x)
       x = x.view(-1,7*7*1024)
       x = self.layer7(x)
       x = x.view(-1,30,7,7)
       return x

class YoloClassifier(Yolo):
    def __init__(self, output_size):
        super(YoloClassifier, self).__init__()
        self.linear = nn.Linear(in_features=4*4*1024,out_features=output_size)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
       x = self.layer1(x)
       x = self.layer2(x)
       x = self.layer3(x)
       x = self.layer4(x)
       x = self.layer5(x)
       x = x.view(-1,4*4*1024)
       x = self.linear(x)
       # x = self.softmax(x)
       x = self.sigmoid(x)
       return x

