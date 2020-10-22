import torch


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp, targ, epsilon=1e-12):
        error = inp - targ
        return torch.mean(torch.log(torch.cosh(error + epsilon)))
