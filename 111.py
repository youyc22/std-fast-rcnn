import numpy as np
import torch
import torch.nn.functional as F

offsets=2*torch.ones(10, 4)
GT_offsets=torch.ones(10, 4)
batch_size=5
a=F.smooth_l1_loss(offsets, GT_offsets, reduction='sum') * 1. / batch_size
print(a)