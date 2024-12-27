# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/12/30 16:56
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
from torch.utils.data import DataLoader

from .ISIC2018Dataset import ISIC2018Dataset
from .MMOTUDataset import MMOTUDataset
from .RevisionDataset import RevisionDataset
from .ToothDataset import ToothDataset
from .KfoldToothDataset import KfoldToothDataset


def get_dataloader(opt, train_images_path_list=None, train_labels_path_list=None, valid_images_path_list=None, valid_labels_path_list=None):
    """
    get dataloader
    Args:
        opt: params dict
        train_images_path_list: images paths of train set
        train_labels_path_list: labels paths of train set
        valid_images_path_list: images paths of valid set
        valid_labels_path_list: labels paths of valid set
    Returns:
    """
    if opt["dataset_name"] == "3D-CBCT-Tooth":
        train_set = ToothDataset(opt, mode="train")
        valid_set = ToothDataset(opt, mode="valid")

        train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    elif opt["dataset_name"] == "Kfold-3D-CBCT-Tooth":
        train_set = KfoldToothDataset(opt, train_images_path_list, train_labels_path_list, mode="train")
        valid_set = KfoldToothDataset(opt, valid_images_path_list, valid_labels_path_list, mode="valid")

        train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    elif opt["dataset_name"] == "MMOTU":
        train_set = MMOTUDataset(opt, mode="train")
        valid_set = MMOTUDataset(opt, mode="valid")

        train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=opt["num_workers"], pin_memory=True)

    elif opt["dataset_name"] == "ISIC-2018":
        train_set = ISIC2018Dataset(opt, mode="train")
        valid_set = ISIC2018Dataset(opt, mode="valid")

        train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=opt["num_workers"], pin_memory=True)
    elif opt["dataset_name"] == "DRIVE" or opt["dataset_name"] == "STARE" or opt["dataset_name"] == "CHASE-DB1" or opt["dataset_name"] == "Kvasir-SEG":
        train_set = RevisionDataset(opt, mode="train")
        valid_set = RevisionDataset(opt, mode="valid")

        train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=opt["num_workers"], pin_memory=True)

    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataloader available")

    opt["steps_per_epoch"] = len(train_loader)

    return train_loader, valid_loader


def get_test_dataloader(opt):
    """
    get test dataloader
    :param opt: params dict
    :return:
    """
    if opt["dataset_name"] == "3D-CBCT-Tooth":
        valid_set = ToothDataset(opt, mode="valid")
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=1, pin_memory=True)

    elif opt["dataset_name"] == "MMOTU":
        valid_set = MMOTUDataset(opt, mode="valid")
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=1, pin_memory=True)

    elif opt["dataset_name"] == "ISIC-2018":
        valid_set = ISIC2018Dataset(opt, mode="valid")
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=1, pin_memory=True)

    elif opt["dataset_name"] == "DRIVE" or opt["dataset_name"] == "STARE" or opt["dataset_name"] == "CHASE-DB1" or opt["dataset_name"] == "Kvasir-SEG":
        valid_set = RevisionDataset(opt, mode="valid")
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=1, pin_memory=True)

    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataloader available")

    return valid_loader
