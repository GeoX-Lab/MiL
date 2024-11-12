import math
from collections import OrderedDict

import torch.nn as nn
import srof.afnn as afnn
from srof.resnet.resblock import afBottleneck, afconv3x3


class afBasicBlock_complex(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, active_f='sigmoid'):
        super(afBasicBlock_complex, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.af1 = afnn.ForgettingLayer([inplanes, 1, 1], MODE=active_f)

        self.relu = nn.ReLU()

        self.conv1_bn_1 = nn.Sequential(OrderedDict([
            ('conv', conv3x3(inplanes, planes, stride)),
            ('bn', nn.BatchNorm2d(planes))
        ]))
        self.conv1_bn_2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(inplanes, planes, stride=stride, kernel_size=1, bias=False)),
            ('bn', nn.BatchNorm2d(planes))
        ]))
        # afconv3x3(inplanes, planes, stride, active_f=active_f)
        self.af2 = afnn.ForgettingLayer([planes, 1, 1], MODE=active_f)

        self.bn2 = nn.BatchNorm2d(planes)
        self.af3 = afnn.ForgettingLayer([planes, 1, 1], MODE=active_f)

        self.conv2_bn_1 = nn.Sequential(OrderedDict([
            ('conv', conv3x3(planes, planes)),
            ('bn', nn.BatchNorm2d(planes))
        ]))
        self.conv2_bn_2 = nn.Sequential(OrderedDict([
            ('conv',
             nn.Conv2d(planes, planes, kernel_size=1, bias=False)),
            ('bn', nn.BatchNorm2d(planes))
        ]))
        self.conv2_identity = nn.BatchNorm2d(planes)
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
            x, att1 = self.af1(x)
            x = self.relu(x)
            x, att2 = self.af2(self.conv1_bn_1(x) + self.conv1_bn_2(x))
            x = self.relu(x)

            x = self.bn2(x)
            x, att3 = self.af3(x)
            x = self.relu(x)

            x = self.conv2_bn_1(x) + self.conv2_bn_2(x) + self.conv2_identity(x)

            if self.downsample is None:
                x += self.stat(identity)
            else:
                x += self.downsample(identity)

            out, att4 = self.af4(x)
            out = self.relu(out)

            if att0 is None:
                return out, [att1, att2, att3, att4]  # , a, a
            else:
                return out, att0 + [att1, att2, att3, att4]
        else:
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv1(x)
            x = self.relu(x)

            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv2(x)

            if self.downsample is None:
                x += self.stat(identity)
            else:
                x += self.downsample(identity)
            x = self.relu(x)
            return x, []

    def reconstruct(self, threshold, ap):
        # prepare att
        att1 = self.af1.att()
        att2 = self.af2.att()
        att3 = self.af3.att()
        att4 = self.af4.att()

        # step1: merge bn1
        w, b = _fuse_bn(self.bn1, self.inplanes)
        weight, bias = _merge_helper(w, b, ap, att1, threshold)
        self.bn1 = nn.Conv2d(weight.size()[0], weight.size()[1], kernel_size=3, padding=1)
        self.bn1.weight = weight
        self.bn1.bias = bias

        # w = nn.Parameter(self.bn1.weight[ap.squeeze() > threshold])
        # b = nn.Parameter(self.bn1.bias[ap.squeeze() > threshold])
        # running_mean = self.bn1.running_mean[ap.squeeze() > threshold]
        # running_var = self.bn1.running_var[ap.squeeze() > threshold]
        # eps = self.bn1.eps
        # self.bn1 = nn.BatchNorm2d(w.size()[0])
        # self.bn1.weight = w
        # self.bn1.bias = b
        # self.bn1.eps = eps
        # self.bn1.running_mean = running_mean
        # self.bn1.running_var = running_var

        # step2: merge conv1
        w1, bias1 = _fuse_bn_conv(self.conv1_bn_1)
        w2, bias2 = _fuse_bn_conv(self.conv1_bn_2)
        w = w1 + _pad_1x1_to_3x3_tensor(w2)
        b = bias1 + bias2
        weight, bias = _merge_helper(w, b, att1, att2, threshold)
        if self.downsample is None:
            self.conv1 = nn.Conv2d(weight.size()[0], weight.size()[1], kernel_size=3, stride=1,
                                   padding=1, bias=True)
        else:
            self.conv1 = nn.Conv2d(weight.size()[0], weight.size()[1], kernel_size=3, stride=2,
                                   padding=1, bias=True)
        self.conv1.weight = weight
        self.conv1.bias = bias

        # step 3: merge bn2
        conv = nn.Conv2d(self.planes, self.planes, kernel_size=1, stride=1)
        weight = torch.unsqueeze(torch.unsqueeze(torch.eye(conv.weight.size()[0]), -1), -1)
        conv.weight = nn.Parameter(weight)
        w, b = _fuse_bn(self.bn2, self.planes)
        weight, bias = _merge_helper(w, b, att2, att3, threshold)
        self.bn2 = nn.Conv2d(weight.size()[0], weight.size()[1], kernel_size=3, padding=1)
        self.bn2.weight = weight
        self.bn2.bias = bias
        w, b = self.bn2.weight, self.bn2.bias
        self.bn2.weight = nn.Parameter(w[att2.squeeze() > threshold])
        self.bn2.bias = nn.Parameter(b[att2.squeeze() > threshold])



        # step 4: merge conv2
        w1, bias1 = _fuse_bn_conv(self.conv2_bn_1)
        w2, bias2 = _fuse_bn_conv(self.conv2_bn_2)
        wi, biasi = _fuse_bn(self.conv2_identity, self.planes)
        w = w1 + _pad_1x1_to_3x3_tensor(w2) + wi
        b = bias1 + bias2 + biasi
        weight, bias = _merge_helper(w, b, att3, att4, threshold)
        self.conv2 = nn.Conv2d(weight.size()[0], weight.size()[1], kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.conv2.weight = weight
        self.conv2.bias = bias

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
        del self.conv2_identity

        return att4

class ResNet50(nn.Module):
    def __init__(self, block, layers, num_classes=1000, active_f='sigmoid'):
        self.inplanes = 16
        super(ResNet50, self).__init__()
        self.num_classes = num_classes
        self.conv1 = afconv3x3(3, self.inplanes,
                               stride=2)  # nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, bias=False,
        #                        padding=3)  # afnn.AfConv2d_bn(3, self.inplanes, kernel_size=3, stride=2,
        #                 bias=False, padding=3)
        # self.bn1 = nn.BatchNorm2d(16) #64
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_flayer(block, 64, layers[0], active_f=active_f)
        self.layer2 = self._make_flayer(block, 128, layers[1], stride=2, active_f=active_f)
        self.layer3 = self._make_flayer(block, 256, layers[2], stride=2, active_f=active_f)
        self.layer4 = self._make_flayer(block, 512, layers[3], stride=2, active_f=active_f)

        self.feature = nn.AdaptiveAvgPool2d((1, 1))  # nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # afnn.AfLinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_flayer(self, block, planes, blocks, stride=1, active_f='sigmoid'):
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        else:
            downsample = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, stride=stride, bias=False)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                downsample=nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, stride=1,
                                                     bias=False), active_f=active_f))
        return nn.Sequential(*layers)

    def forward(self, x):
        x, att = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x, att1 = self.layer1(x)
        x, att2 = self.layer2(x)
        x, att3 = self.layer3(x)
        x, att4 = self.layer4(x)
        x = self.feature(x)
        full = x.view(x.size(0), -1)

        output = self.fc(full)

        att = [att] + att1 + att2 + att3 + att4  # + [att5]

        return output, att  # , W, b


def resnet50_cbam(pretrained=False, **kwargs):
    """atYPICA99Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet50(afBottleneck_complex, [3, 4, 6, 3], **kwargs)
    if pretrained:
        NotImplementedError
    return model
