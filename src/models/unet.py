import torch
import torch.nn as nn
import torch.nn.functional as F

class Basic(nn.Module):
    def __init__(self, ch):
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)
