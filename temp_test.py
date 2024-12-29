# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2024/12/30 01:18
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import torch
import torch.nn.functional as F

# 初始 Tensor
t = torch.tensor([1.0, 2.0, 3.0])
print("初始 Tensor:", t)

# 连续执行 softmax
for i in range(10):
    t = F.softmax(t, dim=0)
    print(f"第 {i + 1} 次 softmax 结果:", t)