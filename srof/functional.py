"""
Basic function of af
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans


def _merge_helper(w, bias, att_pre, att_curr, threshold):
    a = w[att_curr.squeeze() > threshold, :, :, :]
    a = a[:, att_pre.squeeze() > threshold, :, :]
    b = att_curr[:, att_curr.squeeze() > threshold, :, :]
    size = a.size()
    bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])
    if bias != None:
        bias_ = bias[att_curr.squeeze() > threshold]
        return nn.Parameter(torch.mul(a, bw)), nn.Parameter(torch.mul(bias_, b.squeeze()))
    else:
        return nn.Parameter(torch.mul(a, bw)), None


def _bn_cuter(bn, att, threshold):
    """
    for bn pruning(no af, just adjust channel by previous att)

    """
    w = nn.Parameter(bn.weight[att.squeeze() > threshold])
    b = nn.Parameter(bn.bias[att.squeeze() > threshold])
    running_mean = bn.running_mean[att.squeeze() > threshold]
    running_var = bn.running_var[att.squeeze() > threshold]
    eps = bn.eps
    bn.num_features = w.size()[0]
    bn.weight = w
    bn.bias = b
    bn.eps = eps
    bn.running_mean = running_mean
    bn.running_var = running_var
    return bn


def _fuse_bn_conv(branch):
    """
    fuse batchnormal2d and conv3x3
    :param branch: nn.Sequential, which contain 2 layer: conv,bn
    :return: conv weight and bias after fusion

    Attention! bias is needed which have nothing to do with basic conv!
    """
    kernel = branch[0].weight
    running_mean = branch[1].running_mean
    running_var = branch[1].running_var
    gamma = branch[1].weight
    beta = branch[1].bias
    eps = branch[1].eps

    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)

    return kernel * t, beta - running_mean * gamma / std


def _pad_1x1_to_3x3_tensor(kernel1x1):
    """
    channel align, pad smaller one to lager one
    :param kernel1x1: weight of 1x1 conv
    :return: weight of 1x1 conv
    """
    if kernel1x1 is None:
        return 0
    else:
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])


def _merge_weight_output(w, bias, att_pre=None, threshold=None):
    assert threshold is None
    a = w[:, att_pre.squeeze() > threshold]

    return nn.Parameter(a), nn.Parameter(bias), att_pre.squeeze()


def _reconstruct_bn(bn, threshold, att):
    w, bias = _fuse_bn(bn, in_channels=len(bn.bias))

    a = w[att.squeeze() > threshold, :, :, :]
    b = att  # [:, att.squeeze() > threshold, :, :]

    bias_ = bias[att.squeeze() > threshold]
    size = a.size()
    bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])

    return nn.Parameter(torch.mul(a, bw)), nn.Parameter(torch.mul(bias_, b.squeeze()))


def _reconstruct_conv(conv, threshold, att, att_pre=None):
    "without bias"
    if isinstance(conv, nn.Conv2d):
        w = conv.weight
    else:
        w = conv.weight()
    a = w[att.squeeze() > threshold, :, :, :]
    if att_pre is not None:
        a = a[:, att_pre.squeeze() > threshold, :, :]
    b = att[:, att.squeeze() > threshold, :, :]
    size = a.size()
    bw = b.transpose(0, 1).repeat(1, size[1], size[2], size[3])
    if isinstance(conv, nn.Conv2d):
        conv.weight = nn.Parameter(torch.mul(a, bw))
    else:
        conv.set_weight(nn.Parameter(torch.mul(a, bw)))
    return conv


def _fuse_bn(bn, in_channels, groups=1):
    input_dim = in_channels // groups
    kernel_value = np.zeros((in_channels, input_dim, 3, 3), dtype=np.float32)
    for i in range(in_channels):
        kernel_value[i, i % input_dim, 1, 1] = 1
    # self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)

    kernel = torch.from_numpy(kernel_value).to(bn.weight.device)
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


# specially for unequal one
def _merge_conv_bn(conv, bn, att, threshold):
    kernel = conv.weight
    running_mean = bn.running_mean[att.squeeze() > threshold]
    running_var = bn.running_var[att.squeeze() > threshold]
    gamma = bn.weight[att.squeeze() > threshold]
    beta = bn.bias[att.squeeze() > threshold]
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return nn.Parameter(kernel * t), nn.Parameter(beta - running_mean * gamma / std)


def _merge_weight_output(w, bias, att_pre=None, threshold=None):
    # threshold = 0.0005 # 1e-3
    a = w[:, att_pre.squeeze() > threshold]

    return nn.Parameter(a), nn.Parameter(bias), att_pre.squeeze().tolist()


def _pad_1x1_to_3x3_tensor(kernel1x1):
    """
    :param kernel1x1:
    :param t: represent the relationship of branch and main conv
    :return:
    """
    if kernel1x1 is None:
        return 0
    else:
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])


def _pad_1x1_to_7x7_tensor(kernel1x1):
    if kernel1x1 is None:
        return 0
    else:
        return torch.nn.functional.pad(kernel1x1, [3, 3, 3, 3])  # not sure?


def kmeans(att):
    att_ = att.squeeze()
    att_ = att_.cpu().detach().numpy().reshape(-1, 1)
    km = KMeans(n_clusters=2)
    group = km.fit(att_).predict(att_)
    A, B = [], []
    for i in range(len(att_)):
        if group[i] == 0:
            A.append(att_[i])
        else:
            B.append(att_[i])
    # if max(A) > max(B):
    #     print('clustering quality: ' + str(min(A) - max(B)))
    # else:
    #     print('clustering quality: ' + str(min(B) - max(A)))
    return float(min(max(A), max(B))) + 1e-5  # plus a small number to protect model
