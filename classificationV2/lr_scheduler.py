import math

from torch.optim import lr_scheduler


class WarmRestart(lr_scheduler.CosineAnnealingLR):

    def __init__(self, optimizer, T_max=10, T_mult=2, eta_min=1e-8, last_epoch=-1):
        self.T_mult = T_mult
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch == self.T_max:
            self.last_epoch = 0
            self.T_max *= self.T_mult
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]

def warm_restart(scheduler, T_mult=2):
    if scheduler.last_epoch == scheduler.T_max:
        scheduler.last_epoch = -1
        scheduler.T_max *= T_mult
    return scheduler
