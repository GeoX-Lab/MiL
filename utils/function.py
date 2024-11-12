import torch
import numpy as np
import torch.nn as nn


def cal_cov(h):
    shape = list(h.Size())
    mean_h = torch.mean(h)
    mh = torch.mm(mean_h.transpose(), mean_h)  # Not sure about tf.transpose
    vh = torch.mm(h.transpose(), h) / shape[0].type(torch.float)  # Not sure about the Type
    conv_hh = vh - mh
    return conv_hh


def cal_S(r):
    r = np.array(r)
    a1 = np.square(np.sum(r / np.size(r)))
    a2 = np.sum(np.square(r) / np.size(r))
    a3 = (1 - 1 / np.size(r))
    S = (1 - a1 / a2) * a3
    return S


def cal_correlation(num_units):
    ts1 = torch.reshape(torch.range(1, num_units + 1), [num_units, 1])  # watch out the type here(float32)
    ts2 = torch.transpose(ts1)

    ts1 = ts1.repeat(1, num_units)
    ts2 = ts2.repeat(num_units, 1)

    return (num_units - torch.abs(ts1 - ts2)) / (num_units - 1)


# dconv:Frobenius norm of the covariance matrix substract the diagonal
def Deconv(layer):
    cov = cal_cov(layer)
    return torch.sum(cov) - torch.sum(torch.diag(cov))


# cal loss for cross covariance
def cal_cross_cov(h):
    cov = torch.mm(h.transpose(), h)
    return torch.sum(cov) - torch.sum(torch.diag(cov))


# # cal loss for L1
# def unit_l1(layer):
#     return torch.

def lateral_i2(att):
    h_shape = list(att.size())
    if len(h_shape) > 2:
        h1 = att.mean(0).mean(-1).mean(-1).reshape(-1, 1)  # .sum(-1).sum(-1).sum(-1)
        h2 = att.mean(0).mean(-1).mean(-1).reshape(1, -1)  # .sum(-1).sum(-1).sum(-1)
        cov = h1 * h2
        dia = torch.mul(att.mean(0).mean(-1).mean(-1), att.mean(0).mean(-1).mean(-1))  # .sum(-1).sum(-1).sum(-1).sum()
        dia = torch.diag(dia)
        lateral_ = torch.sum(cov - dia) / torch.square(torch.tensor(torch.numel(att))).type(torch.float)  # Not sure
    else:
        h1 = att.mean(0).reshape(-1, 1)
        h2 = att.mean(0).reshape(-1, 1)
        cov = h1 * h2
        dia = torch.mul(att.mean(0), att.mean(0))
        dia = torch.diag(dia)
        lateral_ = torch.sum(cov - dia) / torch.square(torch.tensor(torch.numel(att))).type(torch.float)  # Not sure
    return lateral_


def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)

# def lateral_i2(att, h):
# h_shape = list(h.size())
#
# if len(h_shape) > 2:
#     h1 = torch.mul(h, (1 - att)).mean(0).mean(0).mean(0)  # .sum(-1).sum(-1).sum(-1)
#     h2 = torch.mul(h, att).mean(0).mean(0).mean(0)  # .sum(-1).sum(-1).sum(-1)
#     cov = h1 * h2
#     dia = torch.mul(torch.mul(torch.mul(h, 1 - att), h), att).mean(0)  # .sum(-1).sum(-1).sum(-1).sum()
#     lateral_ = torch.mean(cov - dia) / torch.square(torch.tensor(torch.numel(att))).type(torch.float)  # Not sure
# else:
#     h1 = torch.mul(h, (1 - att)).sum(0)
#     h2 = torch.mul(h, att).sum(0)
#     cov = h1 * h2
#     dia = torch.mul(torch.mul(torch.mul(h, (1 - att)), h), att).sum(0)
#     lateral_ = torch.mean(cov - dia) / torch.square(torch.tensor(torch.numel(att))).type(torch.float)  # Not sure
# return lateral_
