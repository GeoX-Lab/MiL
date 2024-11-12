import copy

from srof.afnn import *
from srof.resnet.resblock import *
from srof.resnet.resnet18 import afBasicBlock
from srof.resnet.resnet50 import afBottleneck
from srof.vggnet.vgg3 import VGG
from utils.config import *


def compress_model(model, backbone, t):
    """
    Reconstruct the model, turn afConv, afLinear into normal conv and linear
    :param model: model which contain the afconv or aflinear
    :param threshold: cut threshold which control the degree of compression. the larger
                     the threshold,the  smaller the number of remaining parameters.
    :return: model after compression
    """
    if backbone == 'resnet18' or backbone == 'resnet18_simple' or backbone == 'resnet18_complex' \
            or backbone == 'resnetTest' or backbone == 'resnet_concat':
        model = compress_resnet18(model, t)
    elif backbone == 'resnet50':
        model = compress_resnet50(model, t)
    elif backbone == 'resnet56' or backbone == 'resnet56_complex' or backbone == 'resnet56_simple':
        model = compress_resnet56(model, t)
    if backbone == 'vgg16' or backbone == 'vgg16_simple':
        model = compress_vgg16(model, t, "cifar10")

    elif backbone == 'vgg3':
        model = compress_vgg3(model)

    return model


def compress_resnet18(model, threshold):
    """
    For ResNet only as it has shortcut.
    :param model: ResNet
    :param threshold:
    :return:
    """
    _model = copy.deepcopy(model)
    _model.eval()

    att = _model.conv1.reconstruct(threshold)
    for m in _model.modules():
        if isinstance(m, afBasicBlock) or isinstance(m, afBasicBlock_simple) \
                or isinstance(m, afBasicBlock_complex):
            att = m.reconstruct(threshold, ap=att)

    weight = _model.fc.weight[:, att.squeeze() > threshold]
    bias = _model.fc.bias
    _model.fc = nn.Linear(512, _model.num_classes)
    _model.fc.weight = nn.Parameter(weight)
    _model.fc.bias = nn.Parameter(bias)

    return _model.to(device)


def compress_resnet50_gdp(model, threshold):
    """
    For ResNet only as it has shortcut.
    :param model: ResNet
    :param model: ResNet
    :param threshold:
    :return:
    """
    _model = copy.deepcopy(model)
    _model.eval()
    # att = _model.conv1.reconstruct(threshold)
    for m in _model.modules():
        if isinstance(m, afBottleneck):
            m.reconstruct(threshold)
    # weight = _model.fc.weight[:, att.squeeze() > threshold]
    # bias = _model.fc.bias
    # _model.fc = nn.Linear(2048, _model.num_classes)
    # _model.fc.weight = nn.Parameter(weight)
    # _model.fc.bias = nn.Parameter(bias)

    return _model.to(device)


def compress_resnet50(model, threshold):
    """
    For ResNet only as it has shortcut.
    :param model: ResNet
    :param model: ResNet
    :param threshold:
    :return:
    """
    _model = copy.deepcopy(model)
    _model.eval()
    att = _model.conv1.reconstruct(threshold)
    for m in _model.modules():
        if isinstance(m, afBottleneck):
            att = m.reconstruct(threshold, att)
    weight = _model.fc.weight[:, att.squeeze() > threshold]
    bias = _model.fc.bias
    _model.fc = nn.Linear(2048, _model.num_classes)
    _model.fc.weight = nn.Parameter(weight)
    _model.fc.bias = nn.Parameter(bias)

    return _model.to(device)


def compress_resnet56(model, threshold):
    """
    For resnet56 only as it has shortcut(same as resnet18)
    :param model:
    :param threshold:
    :return:
    """
    # from srof.resnet.cifarblock import afBasicBlock_complex
    _model = copy.deepcopy(model)
    _model.eval()
    att = _model.conv1.reconstruct(threshold)
    for m in _model.modules():
        if isinstance(m, afBasicBlock_simple) \
                or isinstance(m, afBasicBlock_complex):
            att = m.reconstruct(threshold, ap=att)

    weight = _model.fc.weight[:, att.squeeze() > threshold]
    bias = _model.fc.bias
    _model.fc = nn.Linear(64, _model.num_classes)
    _model.fc.weight = nn.Parameter(weight)
    _model.fc.bias = nn.Parameter(bias)

    return _model.to(device)


def compress_vgg16(model, threshold, datatype):
    """
    For
    :param model:
    :param threshold:
    :return:
    """
    if datatype == 'cifar10':
        times = 1
    else:
        times = 7

    _model = copy.deepcopy(model)
    _model.eval()

    att = None
    for m in _model.features:
        if isinstance(m, AfConv2d_bn) or isinstance(m, AfConv2d_bn_simple):
            att = m.reconstruct(threshold, att_pre=att)

    for m in _model.modules():
        if isinstance(m, AfLinear):
            att = m.reconstruct(threshold, att_pre=att, times=times)
            shape = m._weight().shape[0]

    weight = _model.linear2.weight[:, att.squeeze() > threshold]
    bias = _model.linear2.bias
    _model.linear2 = nn.Linear(shape, _model.num_classes)
    _model.linear2.weight = nn.Parameter(weight)
    _model.linear2.bias = nn.Parameter(bias)

    return _model.to(device)


def compress_vgg3(model):
    threshold = 1
    # for m in model.modules():
    #     if isinstance(m, ForgettingLayer):
    #         t = m.get_max_threshold()
    #         if threshold > t:
    #             threshold = t
    print('Threshold is: ' + str(threshold))

    _model = VGG(model, threshold)

    return _model.to(device)



