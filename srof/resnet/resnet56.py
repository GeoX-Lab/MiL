import math
from srof.resnet.resblock import *


class ResNet56(nn.Module):
    def __init__(self, block, layers, num_classes=1000, active_f='sigmoid'):
        self.inplanes = 16
        super(ResNet56, self).__init__()
        self.num_classes = num_classes
        self.conv1 = afnn.AfConv2d_bn(3, self.inplanes, kernel_size=3, stride=1,
                                      padding=1, active_f=active_f)
        self.layer1 = self._make_flayer(block, 16, layers[0])  # , active_f=active_f)
        self.layer2 = self._make_flayer(block, 32, layers[1], stride=2)  # , active_f=active_f)
        self.layer3 = self._make_flayer(block, 64, layers[2], stride=2)  # , active_f=active_f)

        # self.batch_norm = nn.BatchNorm2d(64)
        self.feature = nn.AdaptiveAvgPool2d((1, 1))  # nn.AvgPool2d(4, stride=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64, num_classes)  # afnn.AfLinear(512 * block.expansion, num_classes)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # for those shortcut, we froze them and init it to equal the input and output
        for pname, m in self.named_modules():
            if 'stat' in pname:
                weight = torch.unsqueeze(torch.unsqueeze(torch.eye(m.weight.size()[0]), -1), -1)
                m.weight = nn.Parameter(weight)
            #

    def _make_flayer(self, block, planes, blocks, stride=1, active_f='sigmoid'):
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes * block.expansion, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        else:
            downsample = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, stride=stride, bias=False)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, active_f))
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

        x, att1 = self.layer1(x)

        x, att2 = self.layer2(x)
        x, att3 = self.layer3(x)


        x = self.relu(x)
        x = self.feature(x)
        full = x.view(x.size(0), -1)

        output = self.fc(full)

        att = [att] + att1 + att2 + att3

        return output, att


def resnet56_cbam(**kwargs):
    """
    Constructs a Resnet-56 model
    :param kwargs:
    :return:
    """
    model = ResNet56(afBasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet56_simple(**kwargs):
    model = ResNet56(afBasicBlock_simple, [9, 9, 9], **kwargs)
    return model


def resnet56_complex(**kwargs):
    model = ResNet56(afBasicBlock_complex, [9, 9, 9], **kwargs)
    return model


def resnet20_complex(**kwargs):
    model = ResNet56(afBasicBlock_complex, [6, 6, 6], **kwargs)
    return model

#
# inpt = torch.randn([1, 16, 32, 32]).to('cuda:0')
# model = resnet56_complex(num_classes=10).to('cuda:0')
# model.eval()
#
# out1, _, _, _ = model.layer1[2](inpt)
#
# ap = torch.ones([1, 16, 1, 1])
# # ap = model.layer1[0].reconstruct(0, ap)
# # ap = model.layer1[1].reconstruct(0, ap)
# model.layer1[2].reconstruct(0, ap)
# # for m in model.layer1.modules():
# #     if isinstance(m, afBasicBlock_complex):
# #         ap = m.reconstruct(0, ap)
# out2, _, _, _ = model.layer1[2](inpt)
#
# print(sum(sum(sum(sum(out2 - out1)))))
# model = resnet56_complex(num_classes=10).to('cuda:0')
# model = compress_resnet56(model, 0)
# stat(model, (3, 32, 32))
