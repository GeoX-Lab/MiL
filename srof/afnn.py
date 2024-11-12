from collections import OrderedDict
import torch.nn as nn

from srof.functional import _fuse_bn_conv, _pad_1x1_to_3x3_tensor, _fuse_bn, \
    _pad_1x1_to_7x7_tensor  # , _pad_1x1_to_7x7_tensor
from utils.config import device
import torch

SIGMA_ini = 0


class ForgettingLayer(nn.Module):
    def __init__(self, shape, MODE):
        super(ForgettingLayer, self).__init__()
        if len(shape) > 2:
            self.params = nn.Parameter(torch.randn([1, shape[0], 1, 1]))  # ([1, 1, 1, shape[0]]))
        else:
            self.params = nn.Parameter(torch.randn([1, shape[0]]))
        self.MODE = MODE  # sigmoid/GDP
        # initialized with MODE
        if MODE == 'sigmoid':
            self.sigma = SIGMA_ini
        elif MODE == 'GDP':
            self.sigma = SIGMA_ini
        elif MODE == 'SSS':
            self.sigma = SIGMA_ini

    def att(self):
        if self.MODE == 'sigmoid':
            return self.sigmoid(self.params, self.sigma)  # self.sigmoid(self.params, self.sigma)
        elif self.MODE == 'GDP':
            return self.polarize(self.params, self.sigma)
        elif self.MODE == 'SSS':
            return self.params

    def sigmoid(self, x, sigma):
        x = 1 / (1 + torch.exp(-x * sigma))
        return x.to(device)  # cuda() if torch.cuda.is_available() else x

    def polarize(self, x, sigma):
        x = x * x / (x * x + sigma)
        return x

    def forward(self, x):
        att = self.att()

        return torch.mul(x, att), att




class AfConv2d_bn(nn.Module):
    """
    AfConv3x3, a block that play the same role of nn.conv2d

    """

    def __init__(self, inchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=False, groups=1, dilation=1,
                 active_f='sigmoid'):
        super(AfConv2d_bn, self).__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias

        if self.inchannels == self.outchannels and self.kernel_size != 1 and self.stride == 1:
            self.identity = True
            self.identity_bn = nn.BatchNorm2d(outchannels)
        else:
            self.identity = False

        if self.kernel_size == 1:
            self.p1 = 0
            self.p2 = 0
        else:
            self.p1 = padding
            self.p2 = padding - (kernel_size - 1) // 2  # int((2 * padding - (kernel_size - 1) * dilation) / 2)

        self.conv_bn_1 = nn.Sequential(OrderedDict([
            ('conv',
             nn.Conv2d(inchannels, outchannels, padding=self.p1, kernel_size=kernel_size, stride=stride, bias=bias,
                       groups=groups, dilation=dilation)),
            ('bn', nn.BatchNorm2d(num_features=outchannels))
        ]))

        self.conv_bn_2 = nn.Sequential(OrderedDict([
            ('conv',
             nn.Conv2d(inchannels, outchannels, padding=self.p2, kernel_size=1, stride=stride, bias=bias)),
            ('bn', nn.BatchNorm2d(num_features=outchannels))
        ]))

        self.af = ForgettingLayer([outchannels, 1, 1], active_f)
        self.conv = None

    def weight(self):
        assert self.conv is not None
        return self.conv.weight

    def bias(self):
        assert self.conv is not None
        return self.conv.bias

    def forward(self, x):
        if self.conv != None:
            return self.conv(x), []

        if self.identity:
            x = self.conv_bn_1(x) + self.conv_bn_2(x) + self.identity_bn(x)
        else:
            x = self.conv_bn_1(x) + self.conv_bn_2(x)
        x = torch.relu(x)
        out, att = self.af(x)

        return out, att

    def reconstruct(self, threshold, att_pre=None):
        """
        :param threshold:
        :param att_pre:
        :return:
        """
        att = self.af.att()

        kernel3x3, bias3x3 = _fuse_bn_conv(self.conv_bn_1)
        kernel1x1, bias1x1 = _fuse_bn_conv(self.conv_bn_2)

        if self.identity:
            kernelid, biasid = _fuse_bn(self.identity_bn, self.inchannels)
            w = kernel3x3 + _pad_1x1_to_3x3_tensor(kernel1x1) + kernelid
            bias = bias1x1 + bias3x3 + biasid
        elif self.kernel_size == 3:
            w = kernel3x3 + _pad_1x1_to_3x3_tensor(kernel1x1)
            bias = bias1x1 + bias3x3
        elif self.kernel_size == 7:
            w = kernel3x3 + _pad_1x1_to_7x7_tensor(kernel1x1)
            bias = bias1x1 + bias3x3
        elif self.kernel_size == 1:
            w = kernel3x3 + kernel1x1
            bias = bias1x1 + bias3x3

        a = w[att.squeeze() > threshold, :, :, :]

        if att_pre is not None:
            a = a[:, att_pre.squeeze() > threshold, :, :]

        b = att[:, att.squeeze() > threshold, :, :]
        bias_ = bias[att.squeeze() > threshold]
        size = a.size()
        bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])

        weight = nn.Parameter(torch.mul(a, bw))

        self.conv = nn.Conv2d(weight.shape[1], weight.shape[0], kernel_size=self.kernel_size, stride=self.stride,
                              padding=self.p1, bias=True)
        self.conv.weight = weight
        self.conv.bias = nn.Parameter(torch.mul(bias_, b.squeeze()))

        del self.conv_bn_1
        del self.conv_bn_2
        del self.af

        return att  # .squeeze()  # .tolist()


class AfConv2d_bn_simple(nn.Module):
    """
    AfConv3x3, a block that play the same role of nn.conv2d
    Test version, without extension(for ablation test)
    """

    def __init__(self, inchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=False, groups=1, dilation=1,
                 active_f='sigmoid'):
        super(AfConv2d_bn_simple, self).__init__()
        self.conv = nn.Conv2d(inchannels, outchannels, padding=padding, kernel_size=kernel_size, stride=stride,
                              bias=bias,
                              groups=groups, dilation=dilation)
        self.bn = nn.BatchNorm2d(num_features=outchannels)
        self.af = ForgettingLayer([outchannels, 1, 1], active_f)

    def weight(self):
        assert self.conv is not None
        return self.conv.weight

    def bias(self):
        assert self.conv is not None
        return self.conv.bias

    def forward(self, x):
        x = self.conv(x)
        if self.af is None:
            return x, []
        else:
            x = self.bn(x)
            out, att = self.af(x)
            return out, att

    def reconstruct(self, threshold, att_pre=None):
        att = self.af.att()
        w, bias = _fuse_bn_conv([self.conv, self.bn])
        a = w[att.squeeze() > threshold, :, :, :]
        if att_pre is not None:
            a = a[:, att_pre.squeeze() > threshold, :, :]

        b = att[:, att.squeeze() > threshold, :, :]
        bias_ = bias[att.squeeze() > threshold]
        size = a.size()
        bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])

        weight = nn.Parameter(torch.mul(a, bw))

        self.conv = nn.Conv2d(weight.shape[1], weight.shape[0], kernel_size=3, stride=1,
                              padding=1, bias=True)
        self.conv.weight = weight
        self.conv.bias = nn.Parameter(torch.mul(bias_, b.squeeze()))

        self.af = None
        return att  # .squeeze()  # .tolist()


class AfLinear(nn.Module):
    """
    AfLinear, play the same role of nn.Linear
    Attention! bias only!
    """

    def __init__(self, infeatures, outfeatures, active_f, bias=True):
        super(AfLinear, self).__init__()
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bias = bias
        self.linear = nn.Linear(infeatures, outfeatures, bias=bias)
        self.af = ForgettingLayer([outfeatures], active_f)

        self.Linear = None

    def _weight(self):
        if self.Linear == None:
            return self.linear.weight
        else:
            return self.Linear.weight

    def _bias(self):
        if self.bias and self.Linear == None:
            return self.linear.bias
        elif self.bias and self.Linear != None:
            return self.Linear.bias
        else:
            return None

    def forward(self, x):
        if self.Linear == None:
            full = self.linear(x)
            h_i, att = self.af(full)
            return h_i, att
        else:
            return self.Linear(x), []

    def reconstruct(self, threshold, att_pre=None, times=1):
        """

        :param threshold:
        :param att_pre:
        :param times: 对于vgg16，特殊的是，图像大小的不同进入linear层的大小不一样，造成linear层的不同，因此在reconstruct的时候策略也不同，这里默认是1
        :return:
        """
        # assert att_pre is None

        att = self.af.att()
        w = self.linear.weight
        bias = self.linear.bias

        if att_pre != None and att_pre.size()[0] == 512:

            att_pre = att_pre.repeat(1, 1, times, times)
            att_pre = att_pre.view(1, -1)
            a = w[att.squeeze() > threshold, :]
            a = a[:, att_pre.squeeze() > threshold]
        elif att_pre != None:
            a = w[att.squeeze() > threshold, :]
            a = a[:, att_pre.squeeze() > threshold]
        else:
            a = w[att.squeeze() > threshold, :]

        b = att[:, att.squeeze() > threshold]
        bias_ = bias[att.squeeze() > threshold]
        size = a.size()
        bw = b.transpose(0, 1).repeat(1, size[1])
        weight = nn.Parameter(torch.mul(a, bw))

        # a = w[:, att_pre.squeeze() > threshold]
        self.Linear = nn.Linear(weight.shape[1], weight.shape[0], bias=self.bias)
        self.Linear.weight = weight
        self.Linear.bias = nn.Parameter(torch.mul(bias_, b.squeeze()))

        del self.linear
        del self.af

        return att.squeeze()  # .tolist()

# test linear code

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# input = torch.randn([1, 25088]).to(device)
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
