import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet56']

from torchstat import stat

from torchsummary import summary


def conv3x3(in_c_out, out_c_out, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_c_out, out_c_out, kernel_size=3, stride=stride, padding=1)


def conv1x1(in_c_out, out_c_out, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_c_out, out_c_out, kernel_size=1, stride=stride)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in, c_out, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(c_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(c_in, c_out, stride)
        self.downsample = downsample
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = conv3x3(c_out, c_out)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, c_in, c_out, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = conv1x1(c_in, c_out)

        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = conv3x3(c_out, c_out, stride)

        self.bn3 = nn.BatchNorm2d(c_out)
        self.conv3 = conv1x1(c_out, c_out * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)

        self.layer1 = self._make_layer(block, 16, 16, layers[0])
        self.layer2 = self._make_layer(block, 16, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, 64, layers[2], stride=2)

        self.batch_norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.1)

    def _make_layer(self, block, c_in, c_out, blocks, stride=1):
        downsample = None
        if c_in != c_out or stride != 1:
            downsample = conv1x1(c_in * block.expansion, c_out * block.expansion, stride)
        layers = []
        layers.append(block(c_in * block.expansion, c_out, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(c_out * block.expansion, c_out))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet56():
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [9, 9, 9])
    return model

# model = resnet56().to('cuda:0')
# stat(model, (3, 32, 32))