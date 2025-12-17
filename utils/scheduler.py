import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler

def get_scheduler(
        scheduler_name, optimizer, milestones, gamma, max_epoch,
        min_lr, warmup_lr_init, warmup_t, warmup_prefix):
    if scheduler_name == 'multi_step_lr':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=milestones, gamma=gamma)
    elif scheduler_name == 'cosine_lr':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=max_epoch)
    elif scheduler_name == 'cosine_lr_warmup':
        scheduler = CosineLRScheduler(
            optimizer=optimizer, t_initial=max_epoch, lr_min=min_lr,
            warmup_lr_init=warmup_lr_init, warmup_t=warmup_t, warmup_prefix=warmup_prefix)
    else:
        raise NotImplementedError

    return scheduler
