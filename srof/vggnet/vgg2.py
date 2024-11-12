"""
This file! test only
"""
import torch.nn as nn
from torch.autograd import Variable
from utils import *


class ForgettingLayer(nn.Module):
    def __init__(self, shape, sigma=10, flags='pixel'):
        super(ForgettingLayer, self).__init__()
        if len(shape) > 2:
            if flags == 'pixel':
                self.params = nn.Parameter(torch.randn([1, shape[0], shape[1], shape[2]]))
            elif flags == 'spatial':
                self.params = nn.Parameter(torch.randn([1, 1, shape[1], shape[2]]))
            elif flags == 'channel':
                self.params = nn.Parameter(torch.randn([1, shape[0], 1, 1]))  # ([1, 1, 1, shape[0]]))
        else:
            self.params = nn.Parameter(torch.randn([1, shape[0]]))
        self.sigma = sigma

    def sigmoid(self, x, sigma):
        x = 1 / (1 + torch.exp(-x * sigma))
        return x.cuda() if torch.cuda.is_available() else x

    def forward(self, x):
        att = self.sigmoid(self.params, self.sigma)
        h_ = torch.mul(x, att)
        return self.params, att, h_


class VGG_I(nn.Module):

    def __init__(self, numclass=100, use_f_layer=True, flags='pixel'):
        super(VGG_I, self).__init__()
        self.use_f_layer = use_f_layer
        # layer 1:
        self.conv_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True)  # to get .weight and .bias
        self.af_1 = ForgettingLayer([32, 32, 32], flags=flags)

        # layer 2:
        self.conv_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
        self.af_2 = ForgettingLayer([32, 32, 32], flags=flags)

        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # layer 3:
        self.identity_3 = nn.Conv2d(32, 64, kernel_size=1, padding=0, bias=True)
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)
        self.af_3 = ForgettingLayer([64, 16, 16], flags=flags)

        # layer 4:
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.af_4 = ForgettingLayer([64, 16, 16], flags=flags)

        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.linear = nn.Linear(4096, 512, bias=True)
        # self.dropout = nn.Dropout()
        self.af_5 = ForgettingLayer([512], flags=flags)

        self.fc = nn.Linear(512, numclass, bias=True)

        self._initialize_weights()

    def forward(self, x):
        h1 = self.conv_1(x)
        if self.use_f_layer:
            w1_i, att1_i, h_1_i = self.af_1(h1)
            h_1_i = torch.relu(h_1_i)
        else:
            h_1_i = torch.relu(h1)

        h2 = self.conv_2(h_1_i)
        if self.use_f_layer:
            w2_i, att2_i, h_2_i = self.af_2(h2 + h_1_i)
            h_2_i = torch.relu(h_2_i)
        else:
            h_2_i = torch.relu(h2)

        pool1 = self.max_pool1(h_2_i)

        h3 = self.conv_3(pool1)
        h3_identity = self.identity_3(pool1)
        if self.use_f_layer:
            w3_i, att3_i, h_3_i = self.af_3(h3 + h3_identity)
            h_3_i = torch.relu(h_3_i)
        else:
            h_3_i = torch.relu(h3)

        h4 = self.conv_4(h_3_i)
        if self.use_f_layer:
            w4_i, att4_i, h_4_i = self.af_4(h4 + h_3_i)
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

        W = [self.conv_1.weight, self.conv_2.weight, self.conv_3.weight, self.conv_4.weight, self.linear.weight]
        b = [self.conv_1.bias, self.conv_2.bias, self.conv_3.bias, self.conv_4.bias, self.linear.bias]

        return output, h, h_i, att, W, b

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
    def __init__(self, model,thresholds):
        super(VGG, self).__init__()

        self.threshold = thresholds
        att_statistic = []

        self.conv_1_weight, self.conv_1_bias, att_list = self._merge_weight_conv([model.conv_1.weight], [model.conv_1.bias],
                                                                            model.af_1.sigmoid(model.af_1.params,
                                                                                               model.af_1.sigma),threshold=self.threshold)
        att_statistic.extend(att_list)

        self.conv_1 = nn.Conv2d(3, self.conv_1_weight.shape[0], kernel_size=3, padding=1, bias=True)
        self.conv_1.weight, self.conv_1.bias = self.conv_1_weight, self.conv_1_bias

        self.conv_2_weight, self.conv_2_bias, att_list = self._merge_weight_conv([model.conv_2.weight], [model.conv_2.bias],
                                                                            model.af_2.sigmoid(model.af_2.params,
                                                                                               model.af_2.sigma),
                                                                            model.af_1.sigmoid(model.af_1.params,
                                                                                               model.af_1.sigma),threshold=self.threshold)
        att_statistic.extend(att_list)

        self.conv_2 = nn.Conv2d(self.conv_2_weight.shape[1], self.conv_2_weight.shape[0], kernel_size=3, padding=1,
                                bias=True)
        self.conv_2.weight, self.conv_2.bias = self.conv_2_weight, self.conv_2_bias

        self.max_pool1 = model.max_pool1

        self.conv_3_weight, self.conv_3_bias, att_list = self._merge_weight_conv([model.conv_3.weight,model.identity_3.weight], [model.conv_3.bias,model.identity_3.bias],
                                                                            model.af_3.sigmoid(model.af_3.params,
                                                                                               model.af_3.sigma),
                                                                            model.af_2.sigmoid(model.af_2.params,
                                                                                               model.af_2.sigma),threshold=self.threshold)
        att_statistic.extend(att_list)

        self.conv_3 = nn.Conv2d(self.conv_3_weight.shape[1], self.conv_3_weight.shape[0], kernel_size=3, padding=1,
                                bias=True)
        self.conv_3.weight, self.conv_3.bias = self.conv_3_weight, self.conv_3_bias

        self.conv_4_weight, self.conv_4_bias, att_list = self._merge_weight_conv([model.conv_4.weight], [model.conv_4.bias],
                                                                            model.af_4.sigmoid(model.af_4.params,
                                                                                               model.af_4.sigma),
                                                                            model.af_3.sigmoid(model.af_3.params,
                                                                                               model.af_3.sigma),threshold=self.threshold)
        att_statistic.extend(att_list)

        self.conv_4 = nn.Conv2d(self.conv_4_weight.shape[1], self.conv_4_weight.shape[0], kernel_size=3, padding=1,
                                bias=True)
        self.conv_4.weight, self.conv_4.bias = self.conv_4_weight, self.conv_4_bias

        self.max_pool2 = model.max_pool1

        self.linear_weight, self.linear_bias, att_list = self._merge_weight_fc(model.linear.weight, model.linear.bias,
                                                                            model.af_5.sigmoid(model.af_5.params,
                                                                                               model.af_5.sigma),
                                                                            model.af_4.sigmoid(model.af_4.params,
                                                                                               model.af_4.sigma),threshold=self.threshold)
        att_statistic.extend(att_list)

        self.linear = nn.Linear(self.linear_weight.shape[1], self.linear_weight.shape[0], bias=True)

        self.linear.weight, self.linear.bias = self.linear_weight, self.linear_bias
        # self.dropout = nn.Dropout()

        self.fc_weight, self.fc_bias, att_list = self._merge_weight_output(model.fc.weight, model.fc.bias,
                                                                       model.af_5.sigmoid(model.af_5.params,
                                                                                          model.af_5.sigma),threshold=self.threshold)
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

        return out, a, a, a, a, a

    def _merge_weight_conv(self, weight_list, bias_list, att_curr, att_pre=None,threshold=None):
        # threshold = 0.0005 # 1e-5
        if len(weight_list) == 1:
            w = weight_list[0]
            bias = bias_list[0]
            if att_pre is None:
                a = w[att_curr.squeeze() > threshold, :, :, :]
                b = att_curr[:, att_curr.squeeze() > threshold, :, :]
                bias_ = bias[att_curr.squeeze() > threshold]

                size = a.size()
                bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])

                return nn.Parameter(torch.mul(a, bw)), nn.Parameter(
                    torch.mul(bias_, b.squeeze())), att_curr.squeeze().tolist()
            else:
                kernel_value = np.zeros((w.shape[1], w.shape[0], 3, 3), dtype=np.float32)
                for i in range(w.shape[1]):
                    kernel_value[i, i % w.shape[0], 1, 1] = 1
                w = w + torch.from_numpy(kernel_value).to(w.device)

                a = w[att_curr.squeeze() > threshold, :, :, :]
                a = a[:, att_pre.squeeze() > threshold, :, :]
                b = att_curr[:, att_curr.squeeze() > threshold, :, :]
                bias_ = bias[att_curr.squeeze() > threshold]

                size = a.size()
                bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])

                return nn.Parameter(torch.mul(a, bw)), nn.Parameter(
                    torch.mul(bias_, b.squeeze())), att_curr.squeeze().tolist()
        else:
            w = weight_list[0] + self._pad_1x1_to_3x3_tensor(weight_list[1])
            bias = bias_list[0] + bias_list[1]
            a = w[att_curr.squeeze() > threshold, :, :, :]
            a = a[:, att_pre.squeeze() > threshold, :, :]
            b = att_curr[:, att_curr.squeeze() > threshold, :, :]
            bias_ = bias[att_curr.squeeze() > threshold]

            size = a.size()
            bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])

            return nn.Parameter(torch.mul(a, bw)), nn.Parameter(
                torch.mul(bias_, b.squeeze())), att_curr.squeeze().tolist()

        # else:
        #     if att_pre is None:
        #         a = w[att_curr.squeeze() > threshold, :]
        #     # elif att_curr is None:
        #     #     a = w[:, att_pre.squeeze() > threshold]
        #     elif len(att_pre.size()) > 2:
        #         att_pre = att_pre.repeat(1, 1, 8, 8)
        #         att_pre = att_pre.view(1, -1)
        #         a = w[att_curr.squeeze() > threshold, :]
        #         a = a[:, att_pre.squeeze() > threshold]
        #
        #     else:
        #         a = w[att_curr.squeeze() > threshold, att_pre.squeeze() > threshold]
        #
        #     b = att_curr[:, att_curr.squeeze() > threshold]
        #     bias_ = bias[att_curr.squeeze() > threshold]
        #     size = a.size()
        #     bw = b.transpose(0, 1).repeat(1, size[1])
        #
        # # write_list(att_list,'result/att_list.txt')


    def _merge_weight(self, w, bias, att_curr, att_pre=None):
        threshold = 0.0005 # 1e-5
        if len(att_curr.size()) > 2:
            if att_pre is None:
                a = w[att_curr.squeeze() > threshold, :, :, :]
            else:
                a = w[att_curr.squeeze() > threshold, :, :, :]
                a = a[:, att_pre.squeeze() > threshold, :, :]

            b = att_curr[:, att_curr.squeeze() > threshold, :, :]
            bias_ = bias[att_curr.squeeze() > threshold]

            size = a.size()
            bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])

        else:
            if att_pre is None:
                a = w[att_curr.squeeze() > threshold, :]
            # elif att_curr is None:
            #     a = w[:, att_pre.squeeze() > threshold]
            elif len(att_pre.size()) > 2:
                att_pre = att_pre.repeat(1, 1, 8, 8)
                att_pre = att_pre.view(1, -1)
                a = w[att_curr.squeeze() > threshold, :]
                a = a[:, att_pre.squeeze() > threshold]

            else:
                a = w[att_curr.squeeze() > threshold, att_pre.squeeze() > threshold]

            b = att_curr[:, att_curr.squeeze() > threshold]
            bias_ = bias[att_curr.squeeze() > threshold]
            size = a.size()
            bw = b.transpose(0, 1).repeat(1, size[1])

        # write_list(att_list,'result/att_list.txt')

        return nn.Parameter(torch.mul(a, bw)), nn.Parameter(torch.mul(bias_, b.squeeze())), att_curr.squeeze().tolist()

    def _merge_weight_fc(self, w, bias, att_curr, att_pre=None,threshold=None):
        # threshold = 0.0005 # 1e-5
        # if len(att_curr.size()) > 2:
        #     if att_pre is None:
        #         a = w[att_curr.squeeze() > threshold, :, :, :]
        #     else:
        #         a = w[att_curr.squeeze() > threshold, :, :, :]
        #         a = a[:, att_pre.squeeze() > threshold, :, :]
        #
        #     b = att_curr[:, att_curr.squeeze() > threshold, :, :]
        #     bias_ = bias[att_curr.squeeze() > threshold]
        #
        #     size = a.size()
        #     bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])

        # else:
        #     if att_pre is None:
        #         a = w[att_curr.squeeze() > threshold, :]
        #     # elif att_curr is None:
        #     #     a = w[:, att_pre.squeeze() > threshold]
        if len(att_pre.size()) > 2:
            att_pre = att_pre.repeat(1, 1, 8, 8)
            att_pre = att_pre.view(1, -1)
            a = w[att_curr.squeeze() > threshold, :]
            a = a[:, att_pre.squeeze() > threshold]

        else:
            a = w[att_curr.squeeze() > threshold, att_pre.squeeze() > threshold]

        b = att_curr[:, att_curr.squeeze() > threshold]
        bias_ = bias[att_curr.squeeze() > threshold]
        size = a.size()
        bw = b.transpose(0, 1).repeat(1, size[1])

        # write_list(att_list,'result/att_list.txt')

        return nn.Parameter(torch.mul(a, bw)), nn.Parameter(torch.mul(bias_, b.squeeze())), att_curr.squeeze().tolist()

    def _merge_weight_output(self, w, bias, att_pre=None,threshold=None):
        # threshold = 0.0005 # 1e-3
        a = w[:, att_pre.squeeze() > threshold]

        return nn.Parameter(a), nn.Parameter(bias), att_pre.squeeze().tolist()

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])