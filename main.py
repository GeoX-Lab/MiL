from AFNet import AFNet
import srof.afnn as afnn
import torch

"""
type: resnet,vgg16,
lr: 0.02,0.001
"""

# repeat_num = 1  # 实验重复次数
epochs = 200  # 200  # 200
lr = 0.052405935504394756  # vgg16:0.001# resnet18:0.02
lamd = 100.56349354210965  # 0.01 # 原来是2
beta = 10.049944895921623  # 0.001

batchsize = 128  # 1024

backbone = 'resnet56_complex'  # _complex'#'resnet_concat'#'resnet56_complex'  # 'resnet18' 'resnet50' 'vgg16' 'vgg' 'mobilenetv2'
use_srof = True
loss = 'adviser_lateral'  # de
dataset = 'cifar10'
active_f = 'sigmoid'
afnn.SIGMA_ini = 3

model = AFNet(batchsize, lr, epochs, backbone=backbone, lamd=lamd, beta=beta, use_srof=use_srof,
              loss=loss, dataset=dataset, active_f=active_f)
# model.beforetrain()
# model.train()
# model.aftertrain()
#
pre = torch.load('compress.pkl', map_location={'cuda:3': 'cuda:0'})
model.finetuning(pre, 250, 0.00000003)
# model.test_model()
