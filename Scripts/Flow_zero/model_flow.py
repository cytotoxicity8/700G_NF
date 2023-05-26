import FrEIA.framework as Ff
import FrEIA.modules as Fm

import torch
import torch.nn as nn
import torch.nn.functional as F

import constant as const
from utils import AllInOneBlock_revised

class AltUB(nn.Module):
    def __init__(self, dim, eps=1e-3):
        super().__init__()
        self.base_mean = nn.Parameter(torch.zeros(1, dim))
        self.base_cov = nn.Parameter(torch.zeros(1, dim))
        self.eps = eps

    def forward(self, z, rev=False):
        if rev:
            return z
        z = (z-self.base_mean)/(torch.exp(self.base_cov))
        return z


def subnet_fc(dims_in, dims_out):
    hidden_channels = int(dims_in * const.RATIO_FC)
    return nn.Sequential(nn.Linear(dims_in, hidden_channels), nn.ReLU(),
                         nn.Linear(hidden_channels,  dims_out))

def subnet_cat(dims_in, dims_out):
    hidden_channels = int(round(dims_in * const.RATIO_CAT))
    return nn.Sequential(nn.Linear(dims_in, hidden_channels), nn.GELU(),
                         nn.Linear(hidden_channels,  dims_out))

def nf_flow(dim, flow_steps, cond_dims):
    nodes = Ff.SequenceINN(dim)
    nodes.append(
            Fm.ConditionalAffineTransform,
            cond = 0,
            cond_shape=(cond_dims, ),
            subnet_constructor = subnet_cat,
            dims_c = [(cond_dims,)],
            clamp_activation = "ATAN", #Originally tanh
            clamp = 2.25 #Originally 2.25
        )
    for i in range(flow_steps):    
        nodes.append(
            AllInOneBlock_revised,
            cond = 0,
            cond_shape = (cond_dims, ),
            subnet_constructor = subnet_fc,
            permute_soft = True,
            gin_block = True,
        )

    return nodes



class CD_Flow(nn.Module):
    def __init__(self, dim_features, flow_steps, cond_dims):
        super().__init__()
        self.nf_flows = nf_flow(dim_features, flow_steps, cond_dims)
        self.base = AltUB(dim_features)
        
    def forward(self, x, c, rev=False):
        loss = 0
        ret = {}
        output, log_jac_dets = self.nf_flows(x, [c, ], rev=rev)
        output = self.base(output, rev=rev)
        loss += torch.mean(
            torch.sum(0.5*(output**2)+self.base.base_cov) - log_jac_dets
        )

        ret['loss'] = loss
        ret['output'] = output

        if not self.training:
            log_prob = -torch.mean(0.5*(output**2) +self.base.base_cov , dim=1)
            prob = torch.exp(log_prob)
            ret['score'] = -prob

        return ret