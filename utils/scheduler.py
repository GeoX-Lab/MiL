import math


class elasticSigmoid:
    def __init__(self, Tmin, Tmax, total_epoch=200):
        super(elasticSigmoid, self).__init__()
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.total_epoch = total_epoch
        self.T = []

    def step(self, epoch):
        step = epoch / self.total_epoch * 20 - 10
        x = 1 / (1 + math.exp(-step * 0.1))  # here sigma= 1 hyper param
        x = self.Tmin + x * (self.Tmax - self.Tmin)
        self.T.append(x)
        return x


class elasticTanh:
    def __init__(self, Tmin, Tmax, total_epoch=200):
        super(elasticTanh, self).__init__()
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.total_epoch = total_epoch
        self.T = []

    def step(self, epoch):
        step = epoch / self.total_epoch * 10 - 5
        x = (1 - math.exp(-2 * step)) / (1 + math.exp(-2 * step))
        x = self.Tmin + (x + 1) * (self.Tmax - self.Tmin) / 2
        self.T.append(x)
        return x


class cosineAnnealingT:
    def __init__(self, Tbase, Tmin, Imax, decay_rate, global_epoch_init=0, total_epoch=200):
        super(cosineAnnealingT, self).__init__()

        self.Tbase = Tbase
        self.Tmin = Tmin
        self.Imax = Imax
        self.Ie = Imax

        self.decay_rate = decay_rate
        self.global_epoch = global_epoch_init
        self.total_epoch = total_epoch
        self.current_epoch = 0

        self.T = []

    def get_T(self):
        T = self.Tmin + (self.Tbase - self.Tmin) * (
                1 + math.cos(math.pi * self.global_epoch / self.Ie)) / 2
        self.T.append(T)
        return T

    def step(self, epoch=None):
        if epoch is not None:
            self.global_epoch = epoch
        self.current_epoch += 1

        self.Tbase = self.get_T()

        if self.current_epoch == int(self.Ie):
            print("restart at epoch {:03d}".format(self.global_epoch + 1))
            self.current_epoch = 0

            self.Ie = self.Ie * self.decay_rate
            self.Imax += self.Ie


#
class StepT:
    def __init__(self, epoch_step=[], T_step=[]):
        if len(epoch_step) != len(T_step):
            print('epoch step must equal to the T step')
            raise ValueError
        self.epoch_step = []
        self.T_step = []
        self.cur = None

    def step(self, epoch):
        if self.cur == None:
            self.cur = self.epoch_step


def cosineTList(epoch, batch):
    Tscheduler = cosineAnnealingT(30, 2, 600, 0.9)
    for i in range(epoch):
        for j in range(batch):
            if j != batch - 1:
                Tscheduler.step()
        Tscheduler.step(i)
    return Tscheduler.T


def elasticTList(epoch):
    # Tscheduler = elasticSigmoid(3, 20, epoch)
    Tscheduler = elasticTanh(3, 40, epoch)
    for i in range(epoch):
        Tscheduler.step(i + 1)
    return Tscheduler.T

# def elasticTList(epoch):
#     Tscheduler = elasticSigmoid(3, 20, epoch)
#     for i in range(epoch):
#         Tscheduler.step(i + 1)
#     return Tscheduler.T
# print(elasticTList(200))
