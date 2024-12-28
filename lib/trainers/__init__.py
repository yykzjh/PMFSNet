# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/12/30 17:02
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
from .isic_2018_trainer import ISIC2018Trainer
from .mmotu_trainer import MMOTUTrainer
from .tooth_trainer import ToothTrainer
from .kfold_tooth_trainer import KfoldToothTrainer


def get_trainer(opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric):
    if opt["dataset_name"] == "3D-CBCT-Tooth":
        trainer = ToothTrainer(opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric)
    elif opt["dataset_name"] == "Kfold-3D-CBCT-Tooth":
        trainer = KfoldToothTrainer(opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric)
    elif opt["dataset_name"] == "MMOTU":
        trainer = MMOTUTrainer(opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric)
    elif opt["dataset_name"] == "ISIC-2018" or opt["dataset_name"] == "DRIVE" or opt["dataset_name"] == "STARE" or opt["dataset_name"] == "CHASE-DB1" or opt["dataset_name"] == "Kvasir-SEG":
        trainer = ISIC2018Trainer(opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric)
    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataset available when initialize trainer")

    return trainer
