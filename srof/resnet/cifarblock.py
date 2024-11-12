"""
ResNet block toolkit1.0
Developed to support shortcut.
"""
from collections import OrderedDict
from srof.functional import _fuse_bn, _fuse_bn_conv, _pad_1x1_to_3x3_tensor, _merge_helper, _bn_cuter
import torch.nn as nn
import srof.afnn as afnn
from utils.config import *
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def afconv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, active_f='sigmoid'):
    "3x3 af convolution with padding and aflayer"
    return afnn.AfConv2d_bn(in_planes, out_planes, kernel_size=3, stride=stride,
                            bias=False, groups=groups, padding=dilation, dilation=dilation, active_f=active_f)


def afconv7x7(in_planes, out_planes, stride=2):
    return afnn.AfConv2d_bn(in_planes, out_planes, kernel_size=7, stride=stride,
                            bias=False, padding=3)


def afconv1x1(in_planes, out_planes, stride=1):
    return afnn.AfConv2d_bn(in_planes, out_planes, kernel_size=1, stride=stride,
                            bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


def linear(infeature, outfeature, bias=True):
    "linear layer"
    return nn.Linear(infeature, outfeature, bias)
    # return afnn.AfLinear(infeature, outfeature, bias)


def aflinear(infeature, outfeature, bias=True):
    "linear layer with aflayer"
    return afnn.AfLinear(infeature, outfeature, bias)


class afBasicBlock_complex(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, active_f='sigmoid'):
        super(afBasicBlock_complex, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)

        self.relu = nn.ReLU()

        self.conv1_bn_1 = conv3x3(inplanes, planes, stride)
        # nn.Sequential(OrderedDict([
        # ('conv', conv3x3(inplanes, planes, stride)),
        # ('bn', nn.BatchNorm2d(planes))
        # ]))
        self.conv1_bn_2 = nn.Conv2d(inplanes, planes, stride=stride, kernel_size=1, bias=False)

        # nn.Sequential(OrderedDict([
        # ('conv', nn.Conv2d(inplanes, planes, stride=stride, kernel_size=1, bias=False)),
        # ('bn', nn.BatchNorm2d(planes))
        # ]))
        # afconv3x3(inplanes, planes, stride, active_f=active_f)
        self.af2 = afnn.ForgettingLayer([planes, 1, 1], MODE=active_f)

        self.bn2 = nn.BatchNorm2d(planes)


        self.conv2_bn_1 = conv3x3(planes, planes)
        # nn.Sequential(OrderedDict([
        # ('conv', conv3x3(planes, planes)),
        # ('bn', nn.BatchNorm2d(planes))
        # ]))
        self.conv2_bn_2 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        # nn.Sequential(OrderedDict([
        # ('conv',
        #  nn.Conv2d(planes, planes, kernel_size=1, bias=False)),
        # ('bn', nn.BatchNorm2d(planes))
        # ]))
        # self.conv2_identity = nn.BatchNorm2d(planes)
        self.af4 = afnn.ForgettingLayer([planes, 1, 1], active_f)

        self.conv1, self.conv2 = None, None
        if isinstance(downsample, nn.Conv2d):
            self.stat = downsample
            self.downsample = None
        else:
            self.stat = None
            self.downsample = downsample

        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

    def forward(self, x):
        if isinstance(x, tuple):
            x, att0 = x[0], x[1]
        else:
            att0 = None

        identity = x

        if self.conv1 is None:
            x = self.bn1(x)
            x = self.relu(x)
            x, att2 = self.af2(self.conv1_bn_1(x) + self.conv1_bn_2(x))
            # x = self.relu(x)

            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv2_bn_1(x) + self.conv2_bn_2(x)  # + self.conv2_identity(x)

            if self.downsample is None:
                x += self.stat(identity)
            else:
                x += self.downsample(identity)

            out, att4 = self.af4(x)
            # out = self.relu(out)

            if att0 is None:
                return out, [att2, att4]  # , a, a
            else:
                return out, att0 + [att2, att4]
        else:
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv1(x)
            # x = self.relu(x)

            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv2(x)

            if self.downsample is None:
                x += self.stat(identity)
            else:
                x += self.downsample(identity)
            # x = self.relu(x)
            return x, []

    def reconstruct(self, threshold, ap):
        # prepare att
        # att1 = self.af1.att()
        att2 = self.af2.att()
        # att3 = self.af3.att()
        att4 = self.af4.att()

        # step1: merge bn1
        self.bn1 = _bn_cuter(self.bn1, ap, threshold)

        # step2: merge conv1
        w1 = self.conv1_bn_1.weight  # _fuse_bn_conv(self.conv1_bn_1)
        w2 = self.conv1_bn_2.weight  # _fuse_bn_conv(self.conv1_bn_2)
        w = w1 + _pad_1x1_to_3x3_tensor(w2)
        # b = bias1 + bias2
        weight, bias = _merge_helper(w, None, ap, att2, threshold)
        if self.downsample is None:
            self.conv1 = nn.Conv2d(weight.size()[0], weight.size()[1], kernel_size=3, stride=1,
                                   padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(weight.size()[0], weight.size()[1], kernel_size=3, stride=2,
                                   padding=1, bias=False)
        self.conv1.weight = weight
        # self.conv1.bias = bias

        # step 3: merge bn2
        self.bn2 = _bn_cuter(self.bn2, att2, threshold)

        # step 4: merge conv2
        w1 = self.conv2_bn_1.weight  # ,self.conv2_bn_1.bias#_fuse_bn_conv(self.conv2_bn_1)
        w2 = self.conv2_bn_2.weight  # _fuse_bn_conv(self.conv2_bn_2)
        # wi, biasi = _fuse_bn(self.conv2_identity, self.planes)
        w = w1 + _pad_1x1_to_3x3_tensor(w2)  # + wi
        # b = bias1 + bias2 + biasi
        weight, bias = _merge_helper(w, None, att2, att4, threshold)
        self.conv2 = nn.Conv2d(weight.size()[0], weight.size()[1], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv2.weight = weight
        # self.conv2.bias = bias

        # step 5: merge downsampler/stat
        if isinstance(self.downsample, nn.Sequential):  # len(self.downsample) == 2:
            w, b = _fuse_bn_conv(self.downsample)  # self.downsample[0].weight  # no bias
            weight, bias = _merge_helper(w, b, ap, att4, threshold)
            self.downsample[0].weight = weight
            self.downsample[0].bias = bias
            self.downsample = self.downsample[0]
        else:
            w = self.stat.weight
            a = w[att4.squeeze() > threshold, :, :, :]
            a = a[:, ap.squeeze() > threshold, :, :]
            b = att4[:, att4.squeeze() > threshold, :, :]
            size = a.size()
            bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])
            self.stat.weight = nn.Parameter(torch.mul(a, bw))

        # recycling resources
        del self.conv1_bn_1
        del self.conv1_bn_2

        del self.conv2_bn_1
        del self.conv2_bn_2
        # del self.conv2_identity

        return att4


# a = torch.randn([1, 16, 32, 32]).to(device)
# downsample = nn.Conv2d(16, 16, kernel_size=1, stride=1, bias=False)
# afcomplex = afBasicBlock_complex(16, 16, stride=1, downsample=downsample, active_f='sigmoid').to(device)
# afcomplex.eval()
# b1, _ = afcomplex(a)
# ap = torch.ones([1, 16, 1, 1])
# afcomplex.reconstruct(0, ap)
# b2, _ = afcomplex(a)
# print(sum(sum(sum(sum(b1 - b2)))))
