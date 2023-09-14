import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLossSoftPlus(nn.Module):
    def __init__(self):
        super(GANLossSoftPlus, self).__init__()

    def __call__(self, input, target_is_real):
        if target_is_real:
            return torch.mean(F.softplus(-input))
        else:
            return torch.mean(F.softplus(input))