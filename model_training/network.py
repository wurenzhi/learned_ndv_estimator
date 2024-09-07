import torch
from torch import nn


class Regressor(nn.Module):
    '''
    A multilayer perception network, see Figure 3
    '''

    def __init__(self, n_in):
        super(Regressor, self).__init__()
        n_mid = int(n_in / 2)
        n_out = 1
        self.net = nn.Sequential(nn.Linear(n_in, n_in),
                                 nn.LeakyReLU(),
                                 nn.Linear(n_in, n_in),
                                 nn.LeakyReLU(),
                                 nn.Linear(n_in, n_in),
                                 nn.LeakyReLU(),
                                 nn.Linear(n_in, n_in),
                                 nn.LeakyReLU(),
                                 nn.Linear(n_in, n_in),
                                 nn.LeakyReLU(),
                                 nn.Linear(n_in, n_mid),
                                 nn.Linear(n_mid, n_out),
                                 nn.ReLU()
                                 )

    def forward(self, data, sample_weight, sample_rate):
        y = -torch.abs(self.net(data.float()))
        ndv_s = torch.exp(data.float()[:, 3]) - 1e-3
        n = torch.exp(data.float()[:, 0]) - 1e-3
        output = (y, sample_rate, ndv_s, n)
        return output


class Loss_gamma_0_6(nn.Module):
    '''
    loss function, see Section 4.3
    '''
    def forward(self, input, target, *args, **kwargs):
        y_pred, r, d_s, n = input
        #d_s = d_s.view(-1, 1)
        l_log_unreduced = torch.square(y_pred - target)
        #l_bound = torch.sqrt(((1 / (r + 1e-20) - 1) / 2 * 0.2554128 + d_s) / (d_s + 1e-6))
        #l_log_bound = torch.square(torch.log(l_bound))
        #l = torch.mean(torch.abs(l_log_unreduced - l_log_bound) + l_log_bound)
        l = torch.mean(l_log_unreduced)
        return l
