import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class BilinearScorer(nn.Module):
    def __init__(self, hidden_dim, role_vocab_size, dropout=0.0, gpu_id=-1):
        super(BilinearScorer, self).__init__()

        if gpu_id > -1:
            self.use_gpu = True
        else:
            self.use_gpu = False
        self.hidden_dim = hidden_dim
        self.role_vocab_size = role_vocab_size

        self.dropout = nn.Dropout(p=dropout)

        self.U = Parameter(
            torch.Tensor(self.hidden_dim, self.role_vocab_size, self.hidden_dim)
        )
        nn.init.orthogonal_(self.U)

        self.bias1 = Parameter(torch.Tensor(1, self.hidden_dim * self.role_vocab_size))
        nn.init.constant_(self.bias1, 0)
        self.bias2 = Parameter(torch.Tensor(1, self.role_vocab_size))
        nn.init.constant_(self.bias2, 0)

    def forward(self, pred_input, args_input):

        b, t, h = pred_input.data.shape
        pred_input = self.dropout(pred_input)
        args_input = self.dropout(args_input)

        first = (
            torch.mm(pred_input.view(-1, h), self.U.view(h, -1)) + self.bias1
        )  # [b*t, h] * [h,r*h] = [b*t,r*h]

        out = torch.bmm(
            first.view(-1, self.role_vocab_size, h), args_input.view(-1, h).unsqueeze(2)
        )  # [b*t,r,h] [b*t, h, 1] = [b*t, r]
        out = out.squeeze(2) + self.bias2
        return out
