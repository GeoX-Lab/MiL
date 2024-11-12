import math
from srof.resnet.resblock import *


class ResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=1000, active_f='sigmoid'):
        self.inplanes = 64
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        # self.conv1 = afconv3x3(3, self.inplanes,
        #                        stride=2)  # nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = afnn.AfConv2d_bn(3, self.inplanes, kernel_size=3, stride=2,
                                      bias=False, padding=3, active_f=active_f)
        # self.bn1 = nn.BatchNorm2d(64)
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
        for pname, m in self.named_modules():
            # for those shortcut, we froze them and init it to equal the input and output
            if ('stat' in pname):
                weight = torch.unsqueeze(torch.unsqueeze(torch.eye(m.weight.size()[0]), -1), -1)
                m.weight = nn.Parameter(weight)

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
        layers.append(block(self.inplanes, planes, stride, downsample, active_f=active_f))
        self.inplanes = planes * block.expansion
        # downsample_plane =
        # ('bn', nn.BatchNorm2d(self.inplanes))

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                downsample=nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, stride=1,
                                                     bias=False), active_f=active_f))
        return nn.Sequential(*layers)

    def forward(self, x):
        x, att = self.conv1(x)
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


def resnet18_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet18(afBasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


def resnet18_simple(**kwargs):
    model = ResNet18(afBasicBlock_simple, [2, 2, 2, 2], **kwargs)
    return model


def resnet18_complex(**kwargs):
    model = ResNet18(afBasicBlock_complex, [2, 2, 2, 2], **kwargs)
    return model

# def resnet18_concat(**kwargs):
#     model = ResNet18(afBasicBlock_concat, [2, 2, 2, 2], **kwargs)
#     return model


# code testing
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# input = torch.randn([1, 3, 32, 32]).to(device)
#
# model = resnet18_complex(num_classes=10).to(device)
# out1, _, _, _ = model(input)
#
# from srof.builder import compress_resnet18
#
# model = compress_resnet18(model, 0)
# out2, _, _, _ = model(input)
# print(out1-out2)
