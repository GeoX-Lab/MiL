# from tensorboardX import SummaryWriter
import nni
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

from srof.afnn import ForgettingLayer
from srof.builder import compress_vgg16, compress_resnet18, compress_resnet50, compress_vgg3, \
    compress_resnet56, compress_model

from utils import *
from thop.profile import profile
from utils.config import *
from utils.optimizer import APGNAG, sgd_optimizer

LOG_MODE = False


class AFNet:
    def __init__(self, batch_size, lr, epochs, backbone='vgg3', lamd=2, beta=20, use_srof=True,
                 loss='adviser_lateral', dataset='cifar10', active_f='sigmoid'):
        self.num_class = DATA_CLS[dataset]
        self.lamd = lamd
        self.beta = beta
        self.batch_size = batch_size
        self.learning_rate = lr
        self.epochs = epochs
        self.exemplar_set = []
        self.use_srof = use_srof
        self.loss_type = loss
        self.active_f = active_f
        # self.great_model = None
        self.backbone = backbone
        self.savefile = 'acc9088/'
        self.dataset = dataset
        self.train_loader, self.test_loader = get_dataset(self.dataset, batch_size)

        return

    def beforetrain(self):
        if self.use_srof == False and self.loss_type == 'adviser_lateral':
            print(
                '!! WARRING !! Original mode not support .adviser_lateral. type loss. We changed it to .cross_entropy.')
            self.loss_type = 'None'
        self.model = get_model(self.backbone, self.use_srof, self.dataset, self.active_f)
        self.model.to(device)

        if self.use_srof:
            all_att = None
            counter = 0
            for m in self.model.modules():
                if isinstance(m, ForgettingLayer):
                    counter += 1
                    att = m.att().squeeze()
                    if all_att == None:
                        all_att = att
                    else:
                        all_att = torch.cat((all_att, att))
                    if LOG_MODE and \
                            (counter == 1 or counter == 5 or counter == 9 or counter == 14):
                        path = os.path.join(self.savefile, str(counter))
                        if not os.path.exists(path):
                            os.makedirs(path)
                        write_list(att.cpu().detach().numpy().tolist(),
                                   os.path.join(path, 'init.txt'))
            file_path = os.path.join(self.savefile, 'att_all_init.png')
            all_att = np.sort(all_att.cpu().detach().numpy()).tolist()
            plot_dense(all_att, file_path)

    def train(self):
        loss_list = []
        acc_list = []
        writer = SummaryWriter(self.savefile)

        weight_param = []
        forget_param = []
        for pname, p in self.model.named_parameters():
            if 'resnet' in self.backbone:
                # We used a conv1x1 to help reparameters in resnet side branch. It was initialized as
                # a diagonal matrix and will not participate parameter update(frozen). In this way,
                # conv1x1(x)=x so the model has same flow as original resnet.
                if 'stat' not in pname:  # or 'af' not in pname:
                    weight_param += [p]

            elif self.active_f == 'SSS':
                if 'af' in pname:
                    forget_param += [p]
                else:
                    weight_param += [p]
            else:
                weight_param += [p]

        opt1 = optim.SGD([{'params': weight_param}], lr=self.learning_rate, momentum=0.9, nesterov=True,
                         weight_decay=0.0005)  # sgd_optimizer(self.model, self.learning_rate,                             0.0005)  #

        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=self.epochs, eta_min=0)

        if self.active_f == 'SSS':
            opt2 = APGNAG([{'params': forget_param}], lr=self.learning_rate * 0.1, momentum=0.9, gamma=0.01)
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=self.epochs)
        else:
            opt2, scheduler2 = None, None

        maxstep = 0  # stupid code
        maxacc = 0
        for epoch in range(self.epochs):
            for step, (images, target) in enumerate(self.train_loader):
                if step > maxstep:
                    maxstep = step
                images, target = images.to(device), target.to(device)
                opt1.zero_grad()
                if opt2 is not None:
                    opt2.zero_grad()
                loss, cost1, cost2, cost3 = self._compute_loss(self.model, images, target, lamd=self.lamd,
                                                               beta=self.beta,
                                                               flags=self.loss_type)
                loss.backward()
                # test mode:
                # for p ,v in self.model.named_parameters():
                #     if 'af' in p:
                #         p.param.grad.data = mask * p.param.grad.data
                #         lasso_grad = p.data * (
                #                 (p.param.data ** 2).sum(dim=(1, 2, 3), keepdim=True) ** (-0.5))
                #         p.param.grad.data.add_(resrep_config.lasso_strength, lasso_grad)
                #

                opt1.step()
                if opt2 is not None:
                    opt2.step()

                print('epoch:%d,step:%d,loss:%.3f,loss1:%.3f,loss2:%.3f,loss3:%.3f' % (epoch, step, loss.item(),
                                                                                       cost1.item(), cost2.item(),
                                                                                       cost3.item()))
                loss_list.append(loss.item())
                writer.add_scalar('loss/TotalLoss', loss, step + epoch * maxstep)
                writer.add_scalar('loss/cost1', cost1, step + epoch * maxstep)
                writer.add_scalar('loss/cost2', cost2, step + epoch * maxstep)
                writer.add_scalar('loss/cost3', cost3, step + epoch * maxstep)
            # self.Tstep(self.model, Tlist.pop(0))
            accuracy = self._test(self.model, self.test_loader)

            # nni.report_intermediate_result(accuracy)
            if scheduler2 is not None:
                scheduler2.step()
            print('epoch:%d, test_accuracy:%3f' % (epoch, accuracy))
            scheduler1.step()

            writer.add_scalar('accuracy', accuracy, epoch)
            if self.use_srof and epoch != (self.epochs - 1):
                self._test_af(writer, epoch)
            else:
                print('epoch:%d,accuracy:%.5f' % (epoch, accuracy))
            acc_list.append(accuracy)

        if not os.path.exists(self.savefile):
            os.makedirs(self.savefile)
        # save acc&loss
        file_path = os.path.join(self.savefile, 'loss_%d.txt' % (self.num_class))
        write_list(loss_list, file_path)
        file_path = os.path.join(self.savefile, 'acc_%d.txt' % (self.num_class))
        write_list(acc_list, file_path)

        file_path = os.path.join(self.savefile, 'loss_af_%d.png' % (self.num_class))
        plot_loss_one_task(loss_list, file_path)
        fig_path = os.path.join(self.savefile, 'acc_af_%d.png' % (self.num_class))
        plot_acc_one_task(acc_list, fig_path)

        return accuracy

    def _test_af(self, writer, epoch):
        counter = 0
        all_att = None
        for m in self.model.modules():
            if isinstance(m, ForgettingLayer):
                # update T
                if self.active_f == 'sigmoid':
                    m.sigma *= 1.018
                if self.active_f == 'GDP':
                    m.sigma *= 0.96
                counter += 1
                att = m.att().squeeze()
                if counter == 1:
                    all_att = att
                else:
                    all_att = torch.cat((all_att, att))
                if LOG_MODE:
                    if counter == 1 or counter == 5 or counter == 9 or counter == 14:  # todo:stupid
                        path = os.path.join(self.savefile, str(counter))
                        if not os.path.exists(path):
                            os.makedirs(path)
                        write_list(att.squeeze().cpu().detach().numpy().tolist(),
                                   os.path.join(path, 'epoch_%d.txt' % epoch))

                writer.add_histogram('att/later' + str(counter), att, epoch)

        if (epoch + 1) % 5 == 0 or epoch == 0:  # (epoch+1) % 20 == 0 or epoch == 0:
            file_path = os.path.join(self.savefile, 'att_all_%d.png' % (epoch + 1))
            # plot_att_hist(file_path, all_att.to('cpu').detach().numpy())
            all_att = np.sort(all_att.squeeze().cpu().detach().numpy()).tolist()
            plot_dense(all_att, file_path)
            if LOG_MODE:
                file_path = os.path.join(self.savefile, 'att_all_%d.txt' % (epoch + 1))
                write_list(all_att, file_path)

    def aftertrain(self):
        filename = self.savefile + '5_increment_%d_net.pkl' % (self.num_class)  # +
        # self.great_model = torch.load(filename)  # , map_location={'cuda:5': 'cuda:0'})
        # print(self._test(self.model, self.test_loader))

        # torch.save(self.model, filename)
        self.model = torch.load(filename, map_location={'cuda:3': 'cuda:0'}).to(device)
        if self.use_srof:
            self._threshold_test(self.model)

    def test_model(self):
        """
        for srof model pruning
        :return:
        """
        filename = self.savefile + '5_increment_%d_net.pkl' % (self.num_class)
        self.model = torch.load(filename)
        # self.model = torch.load('5_increment_10_net.pkl', map_location={'cuda:2': 'cuda:0'})
        self._threshold_test()

    def _threshold_test(self, model):
        att = None
        for m in model.modules():
            if isinstance(m, ForgettingLayer):
                t = m.att()
                t = t.squeeze().view(t.size()[0], -1)
                if att == None:
                    att = t
                else:
                    att = torch.cat((att, t), 1)
        thresholds = np.unique(att.squeeze().cpu().detach().numpy()).tolist()
        # thresholds.sort(reverse=True)  # np.argsort(att_)
        # threshold = max(att)
        accs, params, flops = [], [], []
        # print(len(thresholds))
        save_switch, step = 0, 0
        for t in thresholds:
            t = float(t)
            if t >= 1.0:  # or step % 3 != 0:  # step : 3 (to down sample att)
                acc, pa, fl, model = 0, 0, 0, 0
            else:  #
                step += 1
                cmodel = compress_model(model, self.backbone, t)
                # if step == 1 or save_switch == 0:
                #     torch.save(model, 'compress.pkl')
                #     save_switch = 1
                acc = self._test(cmodel, self.test_loader)
                inputs = torch.randn((1, 3, DATA_SIZE[self.dataset], DATA_SIZE[self.dataset])).to(device)
                fl, pa = profile(cmodel, (inputs,), verbose=False)
                accs.append(acc)
                params.append(pa)
                flops.append(fl)
                print('accuracy:' + str(acc) + '|param:' + str(pa) + "|flops:" + str(fl))
                if step == 1:
                    torch.save(cmodel, 'compress.pkl')

                torch.cuda.empty_cache()

                del cmodel
        return

    def _compute_loss(self, model, imgs, target, lamd=None, beta=None, flags='adviser_lateral'):
        output = model(imgs)
        if isinstance(output, tuple):
            att = output[1]
            output = output[0]

        cost1 = torch.mean(F.cross_entropy(output, target))
        # cost1 += l2_regularization(model, 0.001)

        if flags == 'adviser':
            for i in range(len(att)):
                if i == 0:
                    cost2 = lamd * torch.sum(torch.mul(1 - att[i])) / torch.sum(1 - att[i])
                else:
                    cost2 = cost2 + lamd * torch.sum(torch.mul((1 - att[i], att[i]))) / torch.sum(1 - att[i])
            cost3 = cost2
            cost = cost1 + cost2

        elif flags == 'adviser_lateral':
            for i in range(len(att)):
                if i == 0:
                    cost2 = lamd * torch.sum(torch.mul((1 - att[i]), att[i])) / torch.sum(1 - att[i])
                    cost3 = beta * lateral_i2(att[i])  # h[i]
                else:
                    cost2 = cost2 + lamd * torch.sum(torch.mul((1 - att[i]), att[i])) / torch.sum(1 - att[i])
                    cost3 = cost3 + beta * lateral_i2(att[i])
            cost = cost1 + cost2 + cost3
        else:
            cost = cost1
            cost2 = cost1
            cost3 = cost1
        return cost, cost1, cost2, cost3

    def _test(self, model, testloader, i=None):
        model.eval()
        correct, total = 0.0, 0.0
        for setp, (imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                if self.use_srof:
                    outputs, _ = model(imgs)
                else:
                    outputs = model(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        model.train()
        return accuracy

    def Tstep(self, model, T):
        for m in model.modules():
            if isinstance(m, ForgettingLayer):
                m.sigma = T

    def finetuning(self, model, epochs, lr):
        model.train()
        acc = []
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True,
                              weight_decay=0.0005)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,[60, 80, 120])
        maxacc = 0
        for epoch in range(epochs):
            for step, (images, target) in enumerate(self.train_loader):
                images, target = images.to(device), target.to(device)
                opt.zero_grad()
                loss, _, _, _ = self._compute_loss(model, images, target, flags='cross_entropy')
                loss.backward()
                opt.step()
                print('epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss.item()))
            accuracy = self._test(model, self.test_loader)
            if accuracy > maxacc:
                maxacc = accuracy
                torch.save(model, 'finetuning_%f.pkl' % (accuracy))
            acc.append(accuracy)
            print('epoch:%d, test_accuracy:%3f' % (epoch, accuracy))
            # scheduler.step()
            if epoch == 120:
                opt = torch.optim.SGD(model.parameters(), lr=lr * 0.1, momentum=0.9, nesterov=True,
                                      weight_decay=0.0005)
            if epoch == 210:
                opt = torch.optim.SGD(model.parameters(), lr=lr * 0.01, momentum=0.9, nesterov=True,
                                      weight_decay=0.0005)

        print('max acc:' + str(maxacc))

        write_list(acc, 'finetuning_acc.txt')
        plot_acc_one_task(acc, 'finetuning_acc.png')
