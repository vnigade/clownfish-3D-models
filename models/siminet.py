from torch.functional import F
from torch import nn

__all__ = [
    'SimiNet'
]


class SimiNet(nn.Module):
    def __init__(self, opt):
        super(SimiNet, self).__init__()
        self.output = 1
        self.input = opt.n_finetune_classes * 2
        print("Input", self.input, "Classes", opt.n_finetune_classes)
        N_HIDDEN = 1024
        self.l1 = nn.Linear(self.input, N_HIDDEN, bias=True)
        self.l2 = nn.Linear(N_HIDDEN, N_HIDDEN, bias=True)
        self.l3 = nn.Linear(N_HIDDEN, self.output, bias=True)

    def forward(self, x):
        x = x.view(-1, self.input)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
