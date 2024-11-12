from torchstat import stat
# import torch
from torchsummary import summary
#
# from utils.config import device
#
# model = .to('cpu')
# # model.to('cpu')
# # input = torch.randn((1, 3, 32, 32)).to(device)
#
# outcome = summary(model, input_size=(3, 32, 32))  #
# # a = stat(model, (3, 32, 32))
# print(outcome)

from thop.profile import profile
from torchvision import datasets, transforms

import torch

from backbone import vgg16_bn, resnet18
from backbone.resnet56 import resnet56
from srof import resnet56_complex
from srof.vggnet.vgg_32 import vgg16, vgg16_simple
from srof.builder import compress_resnet18, compress_resnet56, compress_resnet50, compress_vgg16
from utils.config import device, DATA_PATH

#
test_transform = transforms.Compose([
    # transforms.Resize(224),

    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

test_dataset = datasets.CIFAR10(DATA_PATH['cifar10'], train=False, transform=test_transform,
                                download=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100,
                                          shuffle=False,
                                          num_workers=2)


def test(model):
    model.eval()
    correct, total = 0.0, 0.0
    for setp, (imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            outputs, _ = model(imgs)

        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        total += len(labels)
    accuracy = correct.item() / total
    print(accuracy)


def mesure(model):
    inputs = torch.randn((1, 3, 32, 32)).to(device)
    fl, pa = profile(model, (inputs,), verbose=False)
    print('flops:', fl, 'param:', pa)


model = torch.load('compress.pkl', map_location={'cuda:3': 'cuda:0'})
# stat = 0
# for pname, m in model.named_modules():
#     if 'stat' in pname:
#         size = m.weight.size()
#         parm = 1
#         for i in size:
#             parm *= i
#         stat += parm
# print(stat)
mesure(model)
        # weight = torch.unsqueeze(torch.unsqueeze(torch.eye(m.weight.size()[0]), -1), -1)
        # m.weight = nn.Parameter(weight)

# model = torch.load('result/5_increment_10_net.pkl').to(device)
# test(model)
# mesure(model)
# model = compress_resnet56(model,0)
# test(model)
# mesure(model)
# resnet56_cbam().to(device)  # torch.load('result/5_increment:10_net.pkl')  #
# model = compress_resnet18(model, 0)

# model = resnet18_test(num_classes=10)
# print(model)
#
# model.eval()
# correct, total = 0.0, 0.0
# for setp, (imgs, labels) in enumerate(test_loader):
#     imgs, labels = imgs.to(device), labels.to(device)
#     with torch.no_grad():
#         outputs, _, _, _ = model(imgs)
#
#     predicts = torch.max(outputs, dim=1)[1]
#     correct += (predicts.cpu() == labels.cpu()).sum()
#     total += len(labels)
#
# accuracy = correct.item() / total
# print(accuracy)
# torch.load('compress.pkl').to(device)
# model = compress_resnet56(model, 0)


# model = resnet56().to(device)
# mesure(model)
# model1 = resnet18_cbam(num_classes=10).to(device)
# print(model1)
# model = resnet18(num_classes=10).to(device)
# model1 = compress_resnet18(model1, 0)
# model = resnet56().to('cuda:0')
# model = vgg16_bn(num_classes=10).to(device)

# model = vgg16_simple(num_classes=10).to(device)
# model = compress_vgg16(model, 0, 'cifar10')


# summary(model, input_size=(3, 32, 32))
# print(model)
# model2 = resnet18(num_classes=10).to(device)
# stat(model, (3, 32, 32))
