import torch.nn as nn
import torch.nn.functional as F


class MC_Dropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(MC_Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, "
                "but got {}".format(p)
            )
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.dropout(input, p=self.p, training=True, inplace=self.inplace)
