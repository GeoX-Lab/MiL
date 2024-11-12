from AFNet import AFNet
import nni
import srof.afnn as afnn

"""
type: resnet,vgg16,
lr: 0.02,0.001
"""

params = {
    'lr': 0.006,
    'lamd': 10,
    'beta': 5,
    'sigma': 0.5,
}

optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

epochs = 200
batchsize = 128

backbone = 'resnet56_complex'
use_srof = True
loss = 'adviser_lateral'  # de
dataset = 'cifar10'
active_f = 'sigmoid'

afnn.SIGMA_ini = params['sigma']

model = AFNet(batchsize, params['lr'], epochs, backbone=backbone, lamd=params['lamd'], beta=params['beta'],
              use_srof=use_srof, loss=loss, dataset=dataset, active_f=active_f)
model.beforetrain()
acc = model.train()
nni.report_final_result(acc)
model.aftertrain()
