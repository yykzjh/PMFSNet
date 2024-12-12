# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/12/30 17:02
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
from .isic_2018_tester import ISIC2018Tester
from .mmotu_tester import MMOTUTester
from .tooth_tester import ToothTester


def get_tester(opt, model, metrics=None):
    if opt["dataset_name"] == "3D-CBCT-Tooth":
        tester = ToothTester(opt, model, metrics)
    elif opt["dataset_name"] == "MMOTU":
        tester = MMOTUTester(opt, model, metrics)
    elif opt["dataset_name"] == "ISIC-2018" or opt["dataset_name"] == "DRIVE" or opt["dataset_name"] == "STARE" or opt["dataset_name"] == "CHASE-DB1" or opt["dataset_name"] == "Kvasir-SEG":
        tester = ISIC2018Tester(opt, model, metrics)
    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataset available when initialize tester")

    return tester
