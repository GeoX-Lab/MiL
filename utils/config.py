# 一些无关全局变量这里

import torch
import os

device = torch.device("cuda:0")

DATA_SIZE = {'cifar10': 32,
             "imagenet": 224}

DATA_PATH = {'cifar10': '../dataset',  # '/home/geox/project/yedingqi/CMN/CIFAR/CIFAR-10/dataset',
             "imagenet": '../../../../../data1/Classification/ILSVRC2012'}

DATA_CLS = {"cifar10": 10,
            "imagenet": 1000}

# todo: to record all hyper parameter here
config_vgg16 = {
    "lr": 0.006,
    "epoch": 200,
    "lamd": 5,
    "beta": 2,
}
config_resnet56 = {
    "lr": 0.02,
    "epoch": 200,
    "lamd": 200,
    "beta": 20

}
config_resnet18 = {
    "lr": 0.02,
    "epoch": 200,
    "lamd": 200,
    "beta": 20
}
