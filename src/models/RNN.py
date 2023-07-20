import torch
import torch.nn as nn
import torch.nn.functional as F


class rnn(nn.Module):
    def __init__(self):
        super(rnn, self).__init__()
        self.W_reset = None
        self.b_reset = None
        self.W_update = None
        self.b_update = None
        self.W = None
        self.b = None

    def forward(self, h0, Xs):
        h = h0
        for idx, x in enumerate(Xs):
            r = F.relu(self.W_reset.matmul(torch.cat((h, x))) + self.b_reset) # 需要确定dim
            z = F.relu(self.W_update.matmul(torch.cat((h, x))) + self.b_update)

            h_ = h * r
            h_ = F.tanh( self.W.matmul(torch.cat((h_, x))) + self.b )
            
            h = (1-z) * h + z * h_
            pass

        return None