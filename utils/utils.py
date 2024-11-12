import matplotlib
import numpy as np
from torchvision import transforms

from utils.config import DATA_CLS, DATA_PATH

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import srof
import backbone as normal
import torchvision.datasets as datasets
import torch

# sparsity- measurement


'''random prune params'''


# def random_ablation(params, ablation_r):
#     for i in range(len(params)):
#         T_shape =
# plot loss per task
def plot_loss_one_task(list, path):
    iters = range(len(list))

    plt.figure()

    plt.plot(iters, list, 'b', label='training loss')
    plt.title('Training loss')
    plt.xlabel('iters')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(path, dpi=300)
    plt.close()
    # plt.show()


# plot accuracy per task
def plot_acc_one_task(list, path):
    iters = range(len(list))

    plt.figure()

    plt.plot(iters, list, 'g', label='test accuracy')
    plt.title('test accuracy')
    plt.xlabel('iters')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(path, dpi=300)
    plt.close()
    # plt.show()


def plot_att(list, path):
    iters = np.arange(0, 1, 1 / len(list))
    # d = 1/len(list)
    # iters *= d
    plt.figure()

    plt.plot(iters, list, 'b')
    # plt.xlabel('iters')
    # plt.ylabel('accuracy')
    # plt.legend()
    plt.savefig(path, dpi=300)
    plt.close()


# plot sparsity per task
def plot_sparsity_one_task(list, path):
    iters = range(len(list))

    plt.figure()

    plt.plot(iters, list, 'r', label='sparsity')
    plt.title('sparsity')
    plt.xlabel('iters')
    plt.ylabel('S')
    plt.legend()
    plt.savefig(path, dpi=300)
    plt.close()


def write_list(list, file_path):
    filename = open(file_path, 'w')
    for value in list:
        filename.write(str(value) + ',')
    filename.close()


def plot_list(path, xlist, ylist, xlable, ylable, dpi=300):
    plt.figure()

    plt.plot(xlist, ylist, 'r', label=ylable)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.title(ylable)
    plt.legend()
    plt.savefig(path, dpi=dpi)
    plt.close()


def plot_att_hist(path, list, dpi=300):
    plt.figure()

    plt.hist(list, density=True, bins=50)
    plt.xlabel('att')

    plt.savefig(path, dpi=dpi)
    plt.close()


def plot_dense(att, path):
    # for i in range(len(att)):
    #     if i == 0:
    #         att_array = att[i].reshape(1, -1).squeeze().detach().cpu().numpy()
    #     else:
    #         att_array = np.concatenate((att_array,att[i].reshape(1, -1).squeeze().detach().cpu().numpy()))

    n_bins = 1000

    fig, ax = plt.subplots(figsize=(6, 4))

    # plot the cumulative histogram
    n, bins, patches = ax.hist(att, n_bins, density=True, histtype='step',
                               cumulative=True, label='Empirical')
    # tidy up the figure
    # ax.grid(True)
    ax.legend(loc='right')
    # ax.set_title('Cumulative step histograms')
    ax.set_xlabel('threshold of importance (0-1)')
    ax.set_ylabel('density')

    plt.tight_layout()
    plt.savefig(path, dpi=500)
    plt.close()

    # return att_array


def get_model(modelname, use_srof=False, dataset='cifar10', active_f='sigmoid'):
    if use_srof:
        if modelname == 'resnet18':
            model = srof.resnet18_cbam(
                num_classes=10, active_f=active_f)  # VGG_I(numclass=self.num_class, use_f_layer=self.use_f_layer,
        if modelname == 'resnet18_simple':
            model = srof.resnet.resnet18_simple(num_classes=DATA_CLS[dataset], active_f=active_f)
        if modelname == 'resnet18_complex':
            model = srof.resnet.resnet18_complex(num_classes=DATA_CLS[dataset], active_f=active_f)
        if modelname == 'resnet50':
            model = srof.resnet50_cbam(num_classes=DATA_CLS[dataset], active_f=active_f)
        if modelname == 'resnet56':
            model = srof.resnet56_cbam(num_classes=DATA_CLS[dataset], active_f=active_f)
        if modelname == 'resnet56_simple':
            model = srof.resnet56_simple(num_classes=DATA_CLS[dataset], active_f=active_f)
        if modelname == 'resnet56_complex':
            model = srof.resnet56_complex(num_classes=DATA_CLS[dataset], active_f=active_f)
        if modelname == 'vgg3':  # origin srof !
            model = srof.VGG_I(numclass=DATA_CLS[dataset], use_f_layer=True)
        #     flags=self.flag)  # network(VGG(),self.num_class)
        if modelname == 'vgg16' and dataset == 'cifar10':
            from srof.vggnet.vgg_32 import vgg16
            model = vgg16(num_classes=DATA_CLS[dataset], active_f=active_f)
        if modelname == 'vgg16_simple' and dataset == 'cifar10':
            from srof.vggnet.vgg_32 import vgg16_simple
            model = vgg16_simple(num_classes=DATA_CLS[dataset], active_f=active_f)
        if modelname == 'vgg16' and dataset == 'imagenet':
            from srof.vggnet.vgg_224 import vgg16
            model = vgg16(num_classes=DATA_CLS[dataset], active_f=active_f)

    else:
        if modelname == 'resnet18':
            model = normal.resnet18(
                num_classes=10)  # VGG_I(numclass=self.num_class, use_f_layer=self.use_f_layer,
        if modelname == 'resnet50':
            model = normal.resnet50(num_classes=10)
        if modelname == 'resnet56':
            model = normal.resnet56()
        if modelname == 'vgg16_bn':
            model = normal.vgg16_bn(num_classes=DATA_CLS[dataset])
    return model


def get_dataset(dataset, batch_size):
    if dataset == 'cifar10':
        train_transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.RandomCrop((32, 32), padding=4),
            # transforms.Resize(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.24705882352941178),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

        ])

        test_transform = transforms.Compose([
            # transforms.Resize(224),

            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_dataset = datasets.CIFAR10(DATA_PATH[dataset], train=True, transform=train_transform,
                                         download=True)
        test_dataset = datasets.CIFAR10(DATA_PATH[dataset], train=False, transform=test_transform,
                                        download=True)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100,
                                                  shuffle=False,
                                                  num_workers=2)
    else:
        raise NotImplementedError

    return train_loader, test_loader
