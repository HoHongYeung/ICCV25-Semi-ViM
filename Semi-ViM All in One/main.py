#!coding:utf-8
import os
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from utils import datasets
from utils.ramps import exp_warmup
from utils.config import parse_commandline_args
from utils.data_utils import DataSetWarpper
from utils.data_utils import TwoStreamBatchSampler
from utils.data_utils import TransformTwice as twice
from architectures.arch import arch

from trainer import *
build_model = {
    'mtv1': MeanTeacherv1.Trainer,
    'mtv2': MeanTeacherv2.Trainer,
    'piv1': PIv1.Trainer,
    'piv2': PIv2.Trainer,
    'vatv1': VATv1.Trainer,
    'vatv2': VATv2.Trainer,
    'epslab2013v1': ePseudoLabel2013v1.Trainer,
    'epslab2013v2': ePseudoLabel2013v2.Trainer,
    'ipslab2013v1': iPseudoLabel2013v1.Trainer,
    'ipslab2013v2': iPseudoLabel2013v2.Trainer,
    'etempensv1': eTempensv1.Trainer,
    'etempensv2': eTempensv2.Trainer,
    'itempensv1': iTempensv1.Trainer,
    'itempensv2': iTempensv2.Trainer,
    'ictv1': ICTv1.Trainer,
    'ictv2': ICTv2.Trainer,
    'mixmatch': MixMatch.Trainer,
    'semivim': SemiViM.Trainer,
    'efixmatch': eFixMatch.Trainer,
    'emixpslabv1': eMixPseudoLabelv1.Trainer,
    'emixpslabv2': eMixPseudoLabelv2.Trainer,
}

def create_loaders_v1(trainset, evalset, label_idxs, unlab_idxs,
                      num_classes,
                      config):
    if config.data_twice: trainset.transform = twice(trainset.transform)
    if config.data_idxs: trainset = DataSetWarpper(trainset, num_classes)
    ## two-stream batch loader
    batch_size = config.sup_batch_size + config.usp_batch_size
    batch_sampler = TwoStreamBatchSampler(
        unlab_idxs, label_idxs, batch_size, config.sup_batch_size)
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_sampler=batch_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    ## test batch loader
    eval_loader = torch.utils.data.DataLoader(evalset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=2*config.workers,
                                              pin_memory=True,
                                              drop_last=False)
    return train_loader, eval_loader


def create_loaders_v2(trainset, evalset, label_idxs, unlab_idxs,
                      num_classes,
                      config):
    if config.data_twice: trainset.transform = twice(trainset.transform)
    if config.data_idxs: trainset = DataSetWarpper(trainset, num_classes)
    ## supervised batch loader
    label_sampler = SubsetRandomSampler(label_idxs)
    label_batch_sampler = BatchSampler(label_sampler, config.sup_batch_size,
                                       drop_last=True)
    label_loader = torch.utils.data.DataLoader(trainset,
                                          batch_sampler=label_batch_sampler,
                                          num_workers=config.workers,
                                          pin_memory=True)
    ## unsupervised batch loader
    if not config.label_exclude: unlab_idxs += label_idxs
    unlab_sampler = SubsetRandomSampler(unlab_idxs)
    unlab_batch_sampler = BatchSampler(unlab_sampler, config.usp_batch_size,
                                       drop_last=True)
    unlab_loader = torch.utils.data.DataLoader(trainset,
                                          batch_sampler=unlab_batch_sampler,
                                          num_workers=config.workers,
                                          pin_memory=True)
    ## test batch loader
    eval_loader = torch.utils.data.DataLoader(evalset,
                                           batch_size=config.sup_batch_size,
                                           shuffle=False,
                                           num_workers=2*config.workers,
                                           pin_memory=True,
                                           drop_last=False)
    return label_loader, unlab_loader, eval_loader


def create_optim(params, config):
    if config.optim == 'sgd':
        optimizer = optim.SGD(params, config.lr,
                              momentum=config.momentum,
                              weight_decay=config.weight_decay,
                              nesterov=config.nesterov)
    elif config.optim == 'adam':
        optimizer = optim.Adam(params, config.lr)
    return optimizer


def create_lr_scheduler(optimizer, config, total_epochs=None):
    epochs = total_epochs if total_epochs is not None else config.epochs
    if config.lr_scheduler == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=epochs,
                                                   eta_min=config.min_lr)
    elif config.lr_scheduler == 'multistep':
        if config.steps is None: return None
        if isinstance(config.steps, int): config.steps = [config.steps]
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=config.steps,
                                             gamma=config.gamma)
    elif config.lr_scheduler == 'exp-warmup':
        lr_lambda = exp_warmup(config.rampup_length,
                               config.rampdown_length,
                               config.epochs)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif config.lr_scheduler == 'none':
        scheduler = None
    else:
        raise ValueError("No such scheduler: {}".format(config.lr_scheduler))
    return scheduler


def run(config):
    print(config)
    print("pytorch version : {}".format(torch.__version__))
    ## create save directory
    if config.save_freq!=0 and not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    ## prepare data
    dconfig   = datasets.load[config.dataset](config.num_labels)
    if config.model[-1]=='1':
        loaders = create_loaders_v1(**dconfig, config=config)
    elif config.model[-1]=='2' or config.model[-5:]=='match':
        loaders = create_loaders_v2(**dconfig, config=config)
    else:
        raise ValueError('No such model: {}'.format(config.model))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ## prepare architecture
    net = arch[config.arch](dconfig['num_classes'], config.drop_ratio)
    net = net.to(device)
    net2 = arch[config.arch](dconfig['num_classes'], config.drop_ratio)
    net2 = net2.to(device)
    net2.load_state_dict(net.state_dict())

    optimizer = create_optim(net.parameters(), config)
    trainer = build_model[config.model](net, net2, optimizer, device, config)

    label_loader, unlab_loader, eval_loader = loaders

    if config.arm_epochs > 0:
        print("=== Stage A: ARM Pretraining ===")
        trainer.optimizer = create_optim(trainer.model.parameters(), config)
        scheduler = create_lr_scheduler(trainer.optimizer, config, total_epochs=config.arm_epochs)
        if getattr(trainer, "lyapema", None) is not None:
            trainer.lyapema.prev_V = None
        trainer.arm_pretrain(unlab_loader, config.arm_epochs, scheduler=scheduler)

    if config.finetune_epochs > 0:
        print("=== Stage B: Supervised Fine-tuning ===")
        trainer.optimizer = create_optim(trainer.model.parameters(), config)
        scheduler = create_lr_scheduler(trainer.optimizer, config, total_epochs=config.finetune_epochs)
        if getattr(trainer, "lyapema", None) is not None:
            trainer.lyapema.prev_V = None
        trainer.finetune_supervised(label_loader, eval_loader, config.finetune_epochs, scheduler=scheduler)

    if config.semi_epochs > 0:
        print("=== Stage C: Semi-Supervised Training ===")
        trainer.optimizer = create_optim(trainer.model.parameters(), config)
        scheduler = create_lr_scheduler(trainer.optimizer, config, total_epochs=config.semi_epochs)
        if getattr(trainer, "lyapema", None) is not None:
            trainer.lyapema.prev_V = None
        trainer.loop(config.semi_epochs, label_loader, unlab_loader, eval_loader, scheduler=scheduler)


if __name__ == '__main__':
    config = parse_commandline_args()
    run(config)
