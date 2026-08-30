import torch
import torch.nn.functional as F
from model import twohot_loss

def test():
    B = 2
    T = 3
    b_rew = torch.randn(B, T, 1)
    b_rew_shifted = torch.cat([torch.zeros_like(b_rew[:, :1]), b_rew[:, :-1]], dim=1)
    print("b_rew:", b_rew[0, :, 0])
    print("shifted:", b_rew_shifted[0, :, 0])

    b_cont = torch.ones(B, T)
    b_cont[0, 1] = 0.0 # Done at t=1
    b_cont_shifted = torch.cat([torch.ones_like(b_cont[:, :1]), b_cont[:, :-1]], dim=1)
    print("b_cont:", b_cont[0])
    print("shifted:", b_cont_shifted[0])

test()
