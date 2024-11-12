"""
This file! beta version of srof
TEST ONLY
"""
import torch.nn as nn
from torch.autograd import Variable

from srof.afnn import ForgettingLayer
from utils import *




def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class VGG_I(nn.Module):

    def __init__(self, numclass=100, use_f_layer=True):
        super(VGG_I, self).__init__()
        self.use_f_layer = use_f_layer
        # layer 1:
        self.convbn3x3_1 = conv_bn(3, 32, kernel_size=3, stride=1, padding=1)  # to get .weight and .bias
        self.convbn1x1_1 = conv_bn(3, 32, kernel_size=1, stride=1, padding=0)  # to get .weight and .bias
        self.af_1 = ForgettingLayer([32, 32, 32])

        # layer 2:
        self.convbn3x3_2 = conv_bn(32, 32, kernel_size=3, stride=1, padding=1)  # to get .weight and .bias
        self.convbn1x1_2 = conv_bn(32, 32, kernel_size=1, stride=1, padding=0)  # to get .weight and .bias
        self.bn_identity_2 = nn.BatchNorm2d(32)
        self.af_2 = ForgettingLayer([32, 32, 32])

        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # layer 3:
        self.convbn3x3_3 = conv_bn(32, 64, kernel_size=3, stride=1, padding=1)  # to get .weight and .bias
        self.convbn1x1_3 = conv_bn(32, 64, kernel_size=1, stride=1, padding=0)  # to get .weight and .bias
        self.af_3 = ForgettingLayer([64, 16, 16])

        # layer 4:
        self.convbn3x3_4 = conv_bn(64, 64, kernel_size=3, stride=1, padding=1)  # to get .weight and .bias
        self.convbn1x1_4 = conv_bn(64, 64, kernel_size=1, stride=1, padding=0)  # to get .weight and .bias
        self.bn_identity_4 = nn.BatchNorm2d(64)
        self.af_4 = ForgettingLayer([64, 16, 16])

        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.linear = nn.Linear(4096, 512, bias=True)
        # self.dropout = nn.Dropout()
        self.af_5 = ForgettingLayer([512])

        self.fc = nn.Linear(512, numclass, bias=True)

        # self._initialize_weights()

    def forward(self, x):
        h1 = self.convbn3x3_1(x) + self.convbn1x1_1(x)
        if self.use_f_layer:
            w1_i, att1_i, h_1_i = self.af_1(h1)
            h_1_i = torch.relu(h_1_i)
        else:
            h_1_i = torch.relu(h1)

        h2 = self.convbn3x3_2(h_1_i) + self.convbn1x1_2(h_1_i) + self.bn_identity_2(h_1_i)
        if self.use_f_layer:
            w2_i, att2_i, h_2_i = self.af_2(h2)
            h_2_i = torch.relu(h_2_i)
        else:
            h_2_i = torch.relu(h2)

        pool1 = self.max_pool1(h_2_i)

        h3 = self.convbn3x3_3(pool1) + self.convbn1x1_3(pool1)
        if self.use_f_layer:
            w3_i, att3_i, h_3_i = self.af_3(h3)
            h_3_i = torch.relu(h_3_i)
        else:
            h_3_i = torch.relu(h3)

        h4 = self.convbn3x3_4(h_3_i) + self.convbn1x1_4(h_3_i) + self.bn_identity_4(h_3_i)
        if self.use_f_layer:
            w4_i, att4_i, h_4_i = self.af_4(h4)
            h_4_i = torch.relu(h_4_i)
        else:
            h_4_i = torch.relu(h4)

        pool2 = self.max_pool2(h_4_i)

        feature = pool2.view(pool2.size()[0], -1)

        full1 = self.linear(feature)
        # drop1 = self.dropout(full1)
        if self.use_f_layer:
            w5_i, att5_i, h_5_i = self.af_5(full1)
            h_5_i = torch.relu(h_5_i)
        else:
            h_5_i = torch.relu(full1)

        output = self.fc(h_5_i)

        h = [h1, h2, h3, h4, full1]
        h_i = [h_1_i, h_2_i, h_3_i, h_4_i, h_5_i]
        if self.use_f_layer:
            att = [att1_i, att2_i, att3_i, att4_i, att5_i]
        else:
            att = []

        # W = [self.convbn3x3_1, self.convbn1x1_1,
        #      self.convbn3x3_2, self.convbn1x1_2, self.bn_identity_2,
        #      self.convbn3x3_3, self.convbn1x1_3,
        #      self.convbn3x3_4, self.convbn1x1_4, self.bn_identity_4,
        #      self.linear.weight]
        # b = None

        return output, h, h_i, att  # , W, b

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VGG(nn.Module):
    def __init__(self, model, thresholds):
        super(VGG, self).__init__()

        self.threshold = thresholds
        att_statistic = []

        self.conv_1_weight, self.conv_1_bias, att_list = self._merge_weight_conv([model.convbn3x3_1, model.convbn1x1_1],
                                                                                 model.af_1.sigmoid(model.af_1.params,
                                                                                                    model.af_1.sigma),
                                                                                 threshold=self.threshold)
        att_statistic.extend(att_list)

        self.conv_1 = nn.Conv2d(3, self.conv_1_weight.shape[0], kernel_size=3, padding=1, bias=True)
        self.conv_1.weight, self.conv_1.bias = self.conv_1_weight, self.conv_1_bias

        self.conv_2_weight, self.conv_2_bias, att_list = self._merge_weight_conv(
            [model.convbn3x3_2, model.convbn1x1_2, model.bn_identity_2],
            model.af_2.sigmoid(model.af_2.params, model.af_2.sigma),
            model.af_1.sigmoid(model.af_1.params, model.af_1.sigma),
            threshold=self.threshold,
            inchannel=32)
        att_statistic.extend(att_list)

        self.conv_2 = nn.Conv2d(self.conv_2_weight.shape[1], self.conv_2_weight.shape[0], kernel_size=3, padding=1,
                                bias=True)
        self.conv_2.weight, self.conv_2.bias = self.conv_2_weight, self.conv_2_bias

        self.max_pool1 = model.max_pool1

        self.conv_3_weight, self.conv_3_bias, att_list = self._merge_weight_conv([model.convbn3x3_3, model.convbn1x1_3],
                                                                                 model.af_3.sigmoid(model.af_3.params,
                                                                                                    model.af_3.sigma),
                                                                                 model.af_2.sigmoid(model.af_2.params,
                                                                                                    model.af_2.sigma),
                                                                                 threshold=self.threshold)
        att_statistic.extend(att_list)

        self.conv_3 = nn.Conv2d(self.conv_3_weight.shape[1], self.conv_3_weight.shape[0], kernel_size=3, padding=1,
                                bias=True)
        self.conv_3.weight, self.conv_3.bias = self.conv_3_weight, self.conv_3_bias

        self.conv_4_weight, self.conv_4_bias, att_list = self._merge_weight_conv(
            [model.convbn3x3_4, model.convbn1x1_4, model.bn_identity_4],
            model.af_4.sigmoid(model.af_4.params, model.af_4.sigma),
            model.af_3.sigmoid(model.af_3.params, model.af_3.sigma),
            threshold=self.threshold,
            inchannel=64)
        att_statistic.extend(att_list)

        self.conv_4 = nn.Conv2d(self.conv_4_weight.shape[1], self.conv_4_weight.shape[0], kernel_size=3, padding=1,
                                bias=True)
        self.conv_4.weight, self.conv_4.bias = self.conv_4_weight, self.conv_4_bias

        self.max_pool2 = model.max_pool1

        self.linear_weight, self.linear_bias, att_list = self._merge_weight_fc(model.linear.weight, model.linear.bias,
                                                                               model.af_5.sigmoid(model.af_5.params,
                                                                                                  model.af_5.sigma),
                                                                               model.af_4.sigmoid(model.af_4.params,
                                                                                                  model.af_4.sigma),
                                                                               threshold=self.threshold)
        att_statistic.extend(att_list)

        self.linear = nn.Linear(self.linear_weight.shape[1], self.linear_weight.shape[0], bias=True)

        self.linear.weight, self.linear.bias = self.linear_weight, self.linear_bias
        # self.dropout = nn.Dropout()

        self.fc_weight, self.fc_bias, att_list = self._merge_weight_output(model.fc.weight, model.fc.bias,
                                                                           model.af_5.sigmoid(model.af_5.params,
                                                                                              model.af_5.sigma),
                                                                           threshold=self.threshold)
        att_statistic.extend(att_list)

        self.fc = nn.Linear(self.linear_weight.shape[0], 10, bias=True)

        self.fc.weight, self.fc.bias = self.fc_weight, self.fc_bias

    def forward(self, x):
        x = torch.relu(self.conv_1(x))
        x = torch.relu(self.conv_2(x))
        x = self.max_pool1(x)
        x = torch.relu(self.conv_3(x))
        x = torch.relu(self.conv_4(x))
        x = self.max_pool2(x)

        x = x.view(x.size()[0], -1)
        x = torch.relu(self.linear(x))
        # x = self.dropout(x)
        out = self.fc(x)
        a = []

        return out, a, a, a  # , a, a

    def _merge_weight_conv(self, weight_list, att_curr, att_pre=None, threshold=None, inchannel=None):
        if len(weight_list) == 2:
            kernel3x3, bias3x3 = self._fuse_bn_tensor(weight_list[0])
            kernel1x1, bias1x1 = self._fuse_bn_tensor(weight_list[1])
            w = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1)
            bias = bias3x3 + bias1x1
            if att_pre is None:
                a = w[att_curr.squeeze() > threshold, :, :, :]
                b = att_curr[:, att_curr.squeeze() > threshold, :, :]
                bias_ = bias[att_curr.squeeze() > threshold]
                size = a.size()
                bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])
                return nn.Parameter(torch.mul(a, bw)), nn.Parameter(
                    torch.mul(bias_, b.squeeze())), att_curr.squeeze().tolist()
            else:
                a = w[att_curr.squeeze() > threshold, :, :, :]
                a = a[:, att_pre.squeeze() > threshold, :, :]
                b = att_curr[:, att_curr.squeeze() > threshold, :, :]
                bias_ = bias[att_curr.squeeze() > threshold]

                size = a.size()
                bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])

                return nn.Parameter(torch.mul(a, bw)), nn.Parameter(
                    torch.mul(bias_, b.squeeze())), att_curr.squeeze().tolist()
        else:
            kernel3x3, bias3x3 = self._fuse_bn_tensor(weight_list[0])
            kernel1x1, bias1x1 = self._fuse_bn_tensor(weight_list[1])
            kernelid, biasid = self._fuse_bn_tensor(weight_list[2], in_channels=inchannel)
            w = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid
            bias = bias3x3 + bias1x1 + biasid

            a = w[att_curr.squeeze() > threshold, :, :, :]
            a = a[:, att_pre.squeeze() > threshold, :, :]
            b = att_curr[:, att_curr.squeeze() > threshold, :, :]
            bias_ = bias[att_curr.squeeze() > threshold]

            size = a.size()
            bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])

            return nn.Parameter(torch.mul(a, bw)), nn.Parameter(
                torch.mul(bias_, b.squeeze())), att_curr.squeeze().tolist()

    # def _merge_weight(self, w, bias, att_curr, att_pre=None):
    #     threshold = 0.0005  # 1e-5
    #     if len(att_curr.size()) > 2:
    #         if att_pre is None:
    #             a = w[att_curr.squeeze() > threshold, :, :,:]
    #         else:
    #             a = w[att_curr.squeeze() > threshold, :, :, :]
    #             a = a[:, att_pre.squeeze() > threshold, :, :]
    #
    #         b = att_curr[:, att_curr.squeeze() > threshold, :, :]
    #         bias_ = bias[att_curr.squeeze() > threshold]
    #
    #         size = a.size()
    #         bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])
    #
    #     else:
    #         if att_pre is None:
    #             a = w[att_curr.squeeze() > threshold, :]
    #         # elif att_curr is None:
    #         #     a = w[:, att_pre.squeeze() > threshold]
    #         elif len(att_pre.size()) > 2:
    #             att_pre = att_pre.repeat(1, 1, 8, 8)
    #             att_pre = att_pre.view(1, -1)
    #             a = w[att_curr.squeeze() > threshold, :]
    #             a = a[:, att_pre.squeeze() > threshold]
    #
    #         else:
    #             a = w[att_curr.squeeze() > threshold, att_pre.squeeze() > threshold]
    #
    #         b = att_curr[:, att_curr.squeeze() > threshold]
    #         bias_ = bias[att_curr.squeeze() > threshold]
    #         size = a.size()
    #         bw = b.transpose(0, 1).repeat(1, size[1])
    #
    #     # write_list(att_list,'result/att_list.txt')
    #
    #     return nn.Parameter(torch.mul(a, bw)), nn.Parameter(torch.mul(bias_, b.squeeze())), att_curr.squeeze().tolist()

    def _merge_weight_fc(self, w, bias, att_curr, att_pre=None, threshold=None):
        if len(att_pre.size()) > 2:
            att_pre = att_pre.repeat(1, 1, 8, 8)
            att_pre = att_pre.view(1, -1)
            a = w[att_curr.squeeze() > threshold, :]
            a = a[:, att_pre.squeeze() > threshold]

        else:
            a = w[att_curr.squeeze() > threshold, :]
            a = a[:, att_pre.squeeze() > threshold]

        b = att_curr[:, att_curr.squeeze() > threshold]
        bias_ = bias[att_curr.squeeze() > threshold]
        size = a.size()
        bw = b.transpose(0, 1).repeat(1, size[1])

        # write_list(att_list,'result/att_list.txt')

        return nn.Parameter(torch.mul(a, bw)), nn.Parameter(torch.mul(bias_, b.squeeze())), att_curr.squeeze().tolist()

    def _merge_weight_output(self, w, bias, att_pre=None, threshold=None):
        # threshold = 0.0005 # 1e-3
        a = w[:, att_pre.squeeze() > threshold]

        return nn.Parameter(a), nn.Parameter(bias), att_pre.squeeze().tolist()

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch, in_channels=None, groups=1):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)  # , in_channels=None, groups=1)
            if not hasattr(self, 'id_tensor'):
                input_dim = in_channels // groups
                kernel_value = np.zeros((in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                # self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = torch.from_numpy(kernel_value).to(branch.weight.device)
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
