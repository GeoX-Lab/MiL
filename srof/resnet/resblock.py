"""
ResNet block toolkit1.0
Developed to support shortcut.
"""
from collections import OrderedDict
from srof.functional import _fuse_bn, _fuse_bn_conv, _pad_1x1_to_3x3_tensor, _bn_cuter, _merge_helper
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


class afBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, active_f='sigmoid'):
        super(afBasicBlock, self).__init__()
        self.conv1 = afconv3x3(inplanes, planes, stride, active_f=active_f)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn = nn.BatchNorm2d(planes)

        self.downsample = downsample

        self.af = afnn.ForgettingLayer([planes, 1, 1], active_f)
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

    def forward(self, x):
        if isinstance(x, tuple):
            x, att0 = x[0], x[1]
        else:
            att0 = None
        identity = x
        x, att1 = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.bn is not None:  # training stage
            x = self.bn(x)
            identity = self.downsample(identity)
            x = x + identity
            x, att2 = self.af(x)
            out = self.relu(x)
            if att0 is None:
                return out, [att1, att2]  # , a, a
            else:
                return out, att0 + [att1, att2]
        else:  # after reparamter!
            identity = self.downsample(identity)
            x += identity
            out = self.relu(x)
            return out, []

    def reconstruct(self, threshold, ap):
        # step1: turn afconv to normal conv
        att_curr = self.af.att()
        att_conv1 = self.conv1.reconstruct(threshold, ap)  # turn into simple conv
        w, bias = _fuse_bn_conv([self.conv2, self.bn])

        a = w[att_curr.squeeze() > threshold, :, :, :]
        a = a[:, att_conv1.squeeze() > threshold, :, :]
        b = att_curr[:, att_curr.squeeze() > threshold, :, :]

        bias_ = bias[att_curr.squeeze() > threshold]
        size = a.size()
        bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])

        self.conv2.weight = nn.Parameter(torch.mul(a, bw))
        self.conv2.bias = nn.Parameter(torch.mul(bias_, b.squeeze()))

        # step3: reconstruct downsampler by ap and att_curr
        if isinstance(self.downsample, nn.Sequential):
            # w, bias = _fuse_bn_conv(self.downsample)
            w, bias = _fuse_bn_conv(self.downsample)  # self.downsample[0].weight  # no bias
            a = w[att_curr.squeeze() > threshold, :, :, :]
            a = a[:, ap.squeeze() > threshold, :, :]
            b = att_curr[:, att_curr.squeeze() > threshold, :, :]
            bias_ = bias[att_curr.squeeze() > threshold]
            size = a.size()
            bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])
            self.downsample[0].weight = nn.Parameter(torch.mul(a, bw))
            self.downsample[0].bias = nn.Parameter(torch.mul(bias_, b.squeeze()))
            self.downsample = self.downsample[0]
        else:
            w = self.downsample.weight
            a = w[att_curr.squeeze() > threshold, :, :, :]
            a = a[:, ap.squeeze() > threshold, :, :]
            b = att_curr[:, att_curr.squeeze() > threshold, :, :]
            size = a.size()
            bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])
            self.downsample.weight = nn.Parameter(torch.mul(a, bw))

        self.af = None
        self.bn = None

        return att_curr


class afBasicBlock_simple(nn.Module):
    """
    AFBasicBlock without side chain
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, active_f='sigmoid'):
        super(afBasicBlock_simple, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.af1 = afnn.ForgettingLayer([planes, 1, 1], active_f)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

        self.af2 = afnn.ForgettingLayer([planes, 1, 1], active_f)
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

    def forward(self, x):
        if isinstance(x, tuple):
            x, att0 = x[0], x[1]
        else:
            att0 = None
        identity = x

        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
            x, att1 = self.af1(x)
            x = self.relu(x)

            x = self.conv2(x)
            identity = self.downsample(identity)
            x += identity
            x, att2 = self.af2(x)
            out = self.relu(x)
            if att0 is None:
                return out, [att1, att2]
            else:
                return out, att0 + [att1, att2]
        else:
            x = self.relu(x)
            x = self.conv2(x)
            identity = self.downsample(identity)
            x += identity
            out = self.relu(x)

            return out, []

    def reconstruct(self, threshold, ap):
        # step1: dealing with conv1: merge conv1 bn1 and af1
        att_curr = self.af1.att()
        # self.conv1.reconstruct(threshold, ap)  # turn into simple conv
        w, bias = _fuse_bn_conv([self.conv1, self.bn1])

        a = w[att_curr.squeeze() > threshold, :, :, :]
        if ap is not None:
            a = a[:, ap.squeeze() > threshold, :, :]
        b = att_curr[:, att_curr.squeeze() > threshold, :, :]
        bias_ = bias[att_curr.squeeze() > threshold]
        size = a.size()
        bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])
        weight = nn.Parameter(torch.mul(a, bw))
        self.conv1.weight = weight
        self.conv1.bias = nn.Parameter(torch.mul(bias_, b.squeeze()))
        self.bn1 = None
        self.af1 = None

        # step2 : dealing with conv2
        att_conv1 = att_curr
        att_curr = self.af2.att()
        w, bias = _fuse_bn_conv([self.conv2, self.bn2])

        a = w[att_curr.squeeze() > threshold, :, :, :]
        a = a[:, att_conv1.squeeze() > threshold, :, :]

        b = att_curr[:, att_curr.squeeze() > threshold, :, :]
        bias_ = bias[att_curr.squeeze() > threshold]
        size = a.size()
        bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])

        self.conv2.weight = nn.Parameter(torch.mul(a, bw))
        self.conv2.bias = nn.Parameter(torch.mul(bias_, b.squeeze()))

        # step3: reconstruct downsampler by ap and att_curr
        # if isinstance(self.downsample[0], frozenlayer):

        if isinstance(self.downsample, nn.Sequential):  # len(self.downsample) == 2:
            # w, bias = _fuse_bn_conv(self.downsample)
            w, bias = _fuse_bn_conv(self.downsample)  # self.downsample[0].weight  # no bias
            a = w[att_curr.squeeze() > threshold, :, :, :]
            a = a[:, ap.squeeze() > threshold, :, :]
            b = att_curr[:, att_curr.squeeze() > threshold, :, :]
            bias_ = bias[att_curr.squeeze() > threshold]
            size = a.size()
            bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])
            self.downsample[0].weight = nn.Parameter(torch.mul(a, bw))
            self.downsample[0].bias = nn.Parameter(torch.mul(bias_, b.squeeze()))
            self.downsample = self.downsample[0]
        else:
            w = self.downsample.weight
            a = w[att_curr.squeeze() > threshold, :, :, :]
            a = a[:, ap.squeeze() > threshold, :, :]
            b = att_curr[:, att_curr.squeeze() > threshold, :, :]
            size = a.size()
            bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])
            self.downsample.weight = nn.Parameter(torch.mul(a, bw))

        self.af2 = None
        self.bn2 = None

        return att_curr


class afBasicBlock_complex(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, active_f='sigmoid'):
        super(afBasicBlock_complex, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.conv1 = afconv3x3(inplanes, planes, stride, active_f=active_f)
        self.conv1_1 = conv3x3(inplanes, planes, stride)
        self.conv1_2 = nn.Conv2d(inplanes, planes, stride=stride, kernel_size=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.af1 = afnn.ForgettingLayer([planes, 1, 1], MODE=active_f)

        self.conv2_1 = conv3x3(planes, planes)
        # nn.Sequential(OrderedDict([
        # ('conv', conv3x3(planes, planes)),
        # ('bn', nn.BatchNorm2d(planes))
        # ]))
        self.conv2_2 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        # nn.Sequential(OrderedDict([
        # ('conv',
        #  nn.Conv2d(planes, planes, kernel_size=1, bias=False)),
        # ('bn', nn.BatchNorm2d(planes))
        # ]))
        # self.conv2_identity = nn.BatchNorm2d(planes)
        # self.conv2_identity = nn.BatchNorm2d(planes)
        self.conv2 = None

        if isinstance(downsample, nn.Conv2d):
            self.stat = downsample
            self.downsample = None
        else:
            self.stat = None
            self.downsample = downsample

        self.af2 = afnn.ForgettingLayer([planes, 1, 1], active_f)
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

    def forward(self, x):
        if isinstance(x, tuple):
            x, att0 = x[0], x[1]
        else:
            att0 = None
        identity = x

        x = self.bn1(x)
        x = self.relu(x)

        if self.conv2 != None:  # after reparameter
            x = self.conv1(x)

            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv2(x)
            if self.downsample is None:
                x += self.stat(identity)
            else:
                x += self.downsample(identity)
            out = x  # self.relu(x)
            return out, []
        else:
            x = self.conv1_1(x) + self.conv1_2(x)
            x, att1 = self.af1(x)

            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv2_1(x) + self.conv2_2(x)  # + self.conv2_identity(x)
            if self.downsample is None:
                x += self.stat(identity)
            else:
                x += self.downsample(identity)
            x, att2 = self.af2(x)
            out = x  # self.relu(x)
            if att0 is None:
                return out, [att1, att2]  # , a, a
            else:
                return out, att0 + [att1, att2]

    def reconstruct(self, threshold, ap):
        # step1: turn afconv to normal conv
        # att_curr = self.af.att()
        att1 = self.af1.att()
        att2 = self.af2.att()
        self.bn1 = _bn_cuter(self.bn1, att=ap, threshold=threshold)  #
        # # step2: turn conv2 to normal conv

        # # first merged 3 branch into 1
        w1 = self.conv1_1.weight  # _fuse_bn_conv(self.conv2_bn_1)
        w2 = self.conv1_2.weight  # _fuse_bn_conv(self.conv2_bn_2)

        w = w1 + _pad_1x1_to_3x3_tensor(w2)  # + wi
        weight, bias = _merge_helper(w, None, ap, att1, threshold)

        if self.downsample is None:
            self.conv1 = nn.Conv2d(weight.size()[0], weight.size()[1], kernel_size=3, stride=1,
                                   padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(weight.size()[0], weight.size()[1], kernel_size=3, stride=2,
                                   padding=1, bias=False)
        self.conv1.weight = weight

        self.bn2 = _bn_cuter(self.bn2, att=att1, threshold=threshold)

        w1 = self.conv2_1.weight
        w2 = self.conv2_2.weight
        w = w1 + _pad_1x1_to_3x3_tensor(w2)  # + wi
        # b = bias1 + bias2 + biasi
        weight, bias = _merge_helper(w, None, att1, att2, threshold)
        self.conv2 = nn.Conv2d(weight.size()[0], weight.size()[1], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv2.weight = weight

        # recycling resources
        del self.conv2_1, self.conv2_2, self.conv1_1, self.conv1_2

        # # step3: reconstruct downsampler by ap and att_curr
        if isinstance(self.downsample, nn.Sequential):  # len(self.downsample) == 2:
            w, bias = _fuse_bn_conv(self.downsample)  # self.downsample[0].weight  # no bias
            a = w[att2.squeeze() > threshold, :, :, :]
            a = a[:, ap.squeeze() > threshold, :, :]
            b = att2[:, att2.squeeze() > threshold, :, :]
            bias_ = bias[att2.squeeze() > threshold]
            size = a.size()
            bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])
            self.downsample[0].weight = nn.Parameter(torch.mul(a, bw))
            self.downsample[0].bias = nn.Parameter(torch.mul(bias_, b.squeeze()))
            self.downsample = self.downsample[0]
        else:
            w = self.stat.weight
            a = w[att2.squeeze() > threshold, :, :, :]
            a = a[:, ap.squeeze() > threshold, :, :]
            b = att2[:, att2.squeeze() > threshold, :, :]
            size = a.size()
            bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])
            self.stat.weight = nn.Parameter(torch.mul(a, bw))

        return att2


class afBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(afBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = afconv1x1(inplanes, width)
        # self.bn1 = norm_layer(width)
        self.conv2 = afconv3x3(width, width, stride=stride, groups=groups, dilation=dilation)
        # self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.af = afnn.ForgettingLayer([planes * self.expansion, 1, 1])
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if isinstance(x, tuple):
            x, att0 = x[0], x[1]
        else:
            att0 = None
        identity = x
        x, att1 = self.conv1(x)  #
        x = self.relu(x)

        x, att2 = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        if self.bn3 is not None:
            x = self.bn3(x)
            identity = self.downsample(identity)
            x += identity
            x, att3 = self.af(x)
            out = self.relu(x)
            if att0 is None:
                return out, [att1, att2, att3]
            else:
                return out, att0 + [att1, att2, att3]
        else:
            identity = self.downsample(identity)
            x += identity
            out = self.relu(x)
            return out, []

    def reconstruct(self, threshold, ap):
        att_curr = self.af.att()
        att_conv1x1_1 = self.conv1.reconstruct(threshold, ap)  # turn into simple conv
        att_conv3x3_2 = self.conv2.reconstruct(threshold, att_conv1x1_1)

        w, bias = _fuse_bn_conv([self.conv3, self.bn3])

        a = w[att_curr.squeeze() > threshold, :, :, :]
        a = a[:, att_conv3x3_2.squeeze() > threshold, :, :]
        b = att_curr[:, att_curr.squeeze() > threshold, :, :]

        bias_ = bias[att_curr.squeeze() > threshold]
        size = a.size()
        bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])

        weight = nn.Parameter(torch.mul(a, bw))
        bias = nn.Parameter(torch.mul(bias_, b.squeeze()))

        self.conv3 = conv1x1(weight.shape[1], weight.shape[0])
        self.conv3.weight = weight
        self.conv3.bias = bias
        # step3: reconstruct downsampler by ap and att_curr
        if isinstance(self.downsample, nn.Sequential):  # len(self.downsample) == 2:
            # w, bias = _fuse_bn_conv(self.downsample)
            w, bias = _fuse_bn_conv(self.downsample)  # self.downsample[0].weight  # no bias
            a = w[att_curr.squeeze() > threshold, :, :, :]
            a = a[:, ap.squeeze() > threshold, :, :]
            b = att_curr[:, att_curr.squeeze() > threshold, :, :]
            bias_ = bias[att_curr.squeeze() > threshold]
            size = a.size()
            bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])
            self.downsample[0].weight = nn.Parameter(torch.mul(a, bw))
            self.downsample[0].bias = nn.Parameter(torch.mul(bias_, b.squeeze()))
            self.downsample = self.downsample[0]
        else:
            w = self.downsample.weight
            a = w[att_curr.squeeze() > threshold, :, :, :]
            a = a[:, ap.squeeze() > threshold, :, :]
            b = att_curr[:, att_curr.squeeze() > threshold, :, :]
            size = a.size()
            bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])
            self.downsample.weight = nn.Parameter(torch.mul(a, bw))

        self.af = None
        self.bn3 = None

        return att_curr


class afBottleneck_complex(nn.Module):
    """
    Ket problem: Should we put 1x1 with 3x3 side chain?
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, active_f='sigmoid'):
        super(afBottleneck_complex, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1_bn_1 = nn.Sequential(OrderedDict([
            ('conv',
             nn.Conv2d(inplanes, width, padding=1, kernel_size=3, bias=False)),
            ('bn', nn.BatchNorm2d(num_features=width))
        ]))

        self.conv1_bn_2 = nn.Sequential(OrderedDict([
            ('conv',
             nn.Conv2d(inplanes, width, padding=0, kernel_size=1, bias=False)),
            ('bn', nn.BatchNorm2d(num_features=width))
        ]))  # conv1x1(width, planes * self.expansion)#afconv1x1(inplanes, width)
        self.af1 = afnn.ForgettingLayer([width, 1, 1], active_f)

        self.conv2 = afconv3x3(width, width, stride=stride, groups=groups, dilation=dilation, active_f=active_f)

        self.conv3_bn_1 = nn.Sequential(OrderedDict([
            ('conv',
             nn.Conv2d(width, planes * self.expansion, padding=1, kernel_size=3, bias=False)),
            ('bn', nn.BatchNorm2d(num_features=planes * self.expansion))
        ]))

        self.conv3_bn_2 = nn.Sequential(OrderedDict([
            ('conv',
             nn.Conv2d(width, planes * self.expansion, padding=0, kernel_size=1, bias=False)),
            ('bn', nn.BatchNorm2d(num_features=planes * self.expansion))
        ]))  # conv1x1(width, planes * self.expansion)
        # self.identity_bn = nn.BatchNorm2d(planes * self.expansion)
        self.af3 = afnn.ForgettingLayer([planes * self.expansion, 1, 1], active_f)

        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.conv1 = None
        self.conv3 = None

    def forward(self, x):
        if isinstance(x, tuple):
            x, att0 = x[0], x[1]
        else:
            att0 = None
        identity = x

        if self.conv1 is None:
            x = self.conv1_bn_1(x) + self.conv1_bn_2(x)
            x, att1 = self.af1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.relu(x)

        # x, att1 = self.conv1(x)  #
        # x = self.relu(x)
        x, att2 = self.conv2(x)
        x = self.relu(x)

        if self.conv3 is None:
            x = self.conv3_bn_1(x) + self.conv3_bn_2(x)  # + self.identity_bn(x)
            identity = self.downsample(identity)
            x += identity
            x, att3 = self.af3(x)
            out = self.relu(x)
            if att0 is None:
                return out, [att1, att2, att3]
            else:
                return out, att0 + [att1, att2, att3]
        else:
            x = self.conv3(x)
            identity = self.downsample(identity)
            x += identity
            out = self.relu(x)
            return out, []

    def reconstruct(self, threshold, ap):
        att_conv1x1_1 = self.af1.att()
        att_conv1x1_3 = self.af3.att()
        # att_conv1x1_1 = self.conv1.reconstruct(threshold, ap)  # turn into simple conv
        w1, bias1 = _fuse_bn_conv(self.conv1_bn_1)
        w2, bias2 = _fuse_bn_conv(self.conv1_bn_2)

        w = w1 + w2
        bias = bias1 + bias2
        a = w[att_conv1x1_1.squeeze() > threshold, :, :, :]
        a = a[:, ap.squeeze() > threshold, :, :]
        b = att_conv1x1_1[:, att_conv1x1_1.squeeze() > threshold, :, :]

        bias_ = bias[att_conv1x1_1.squeeze() > threshold]
        size = a.size()
        bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])

        weight = nn.Parameter(torch.mul(a, bw))
        bias = nn.Parameter(torch.mul(bias_, b.squeeze()))
        self.conv1 = conv1x1(weight.shape[1], weight.shape[0])
        self.conv1.weight = weight
        self.conv1.bias = bias

        att_conv3x3_2 = self.conv2.reconstruct(threshold, att_conv1x1_1)

        # step2: merge conv3
        w1, bias1 = _fuse_bn_conv(self.conv3_bn_1)
        w2, bias2 = _fuse_bn_conv(self.conv3_bn_2)
        # kid, bid = _fuse_bn(self.identity_bn)

        w = w1 + w2  # + kid
        bias = bias1 + bias2  # + bid

        a = w[att_conv1x1_3.squeeze() > threshold, :, :, :]
        a = a[:, att_conv3x3_2.squeeze() > threshold, :, :]
        b = att_conv1x1_3[:, att_conv1x1_3.squeeze() > threshold, :, :]

        bias_ = bias[att_conv1x1_3.squeeze() > threshold]
        size = a.size()
        bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])

        weight = nn.Parameter(torch.mul(a, bw))
        bias = nn.Parameter(torch.mul(bias_, b.squeeze()))

        self.conv3 = conv1x1(weight.shape[1], weight.shape[0])
        self.conv3.weight = weight
        self.conv3.bias = bias
        # step3: reconstruct downsampler by ap and att_curr
        if isinstance(self.downsample, nn.Sequential):  # len(self.downsample) == 2:
            # w, bias = _fuse_bn_conv(self.downsample)
            w, bias = _fuse_bn_conv(self.downsample)  # self.downsample[0].weight  # no bias
            a = w[att_conv1x1_3.squeeze() > threshold, :, :, :]
            a = a[:, ap.squeeze() > threshold, :, :]
            b = att_conv1x1_3[:, att_conv1x1_3.squeeze() > threshold, :, :]
            bias_ = bias[att_conv1x1_3.squeeze() > threshold]
            size = a.size()
            bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])
            self.downsample[0].weight = nn.Parameter(torch.mul(a, bw))
            self.downsample[0].bias = nn.Parameter(torch.mul(bias_, b.squeeze()))
            self.downsample = self.downsample[0]
        else:
            w = self.downsample.weight
            a = w[att_conv1x1_3.squeeze() > threshold, :, :, :]
            a = a[:, ap.squeeze() > threshold, :, :]
            b = att_conv1x1_3[:, att_conv1x1_3.squeeze() > threshold, :, :]
            size = a.size()
            bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])
            self.downsample.weight = nn.Parameter(torch.mul(a, bw))

        self.af1, self.af3 = None, None

        return att_conv1x1_3

# code testing!
#
# super_linear = AfLinear(512 * 7 * 7, 4096).to(device)
# super_linear.eval()
#
# super_out = super_linear(input)
#
# super_linear.reconstruct(0)
# sub_out = super_linear(input)
#
# print(torch.sum(torch.abs(torch.relu(super_out[-1]) - torch.relu(sub_out[-1]))))
# print('finished')

# test conv code

# input = torch.randn([1,3,32,32]).to(device)
#
# super_inconv = AfConv2d_bn(3, 64, kernel_size=1, stride=2,
#                             bias=False, padding=1).to(device)
#
# super_inconv.eval()
# super_out = super_inconv(input)
#
# super_inconv.reconstruct(0)
#
# sub_out = super_inconv(input)
#
# print(torch.sum(torch.abs(torch.relu(super_out[-1]) - torch.relu(sub_out[-1]))))
#
# print('finished')
# a = torch.randn([1, 16, 32, 32]).to(device)
# downsample = nn.Conv2d(16, 16, kernel_size=1, stride=1, bias=False)
# afcomplex = afBasicBlock_complex(16, 16, stride=1, downsample=downsample, active_f='sigmoid').to(device)
# afcomplex.eval()
# b1, _ = afcomplex(a)
# ap = torch.ones([1, 16, 1, 1])
# afcomplex.reconstruct(0, ap)
# b2, _ = afcomplex(a)
# print(sum(sum(sum(sum(b1 - b2)))))
