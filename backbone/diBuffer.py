import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from backbone.gumbel_softmax import gumbel_softmax
from backbone.adain import calc_mean_std, adaptive_instance_normalization_meanstd

class diBuffer(nn.Module):
    def __init__(self, feat_dim:int, buffer_dim:int, buffer_size=200):
        super(diBuffer, self).__init__()

        self.k = nn.Linear(feat_dim, buffer_size, bias=False)
        self.v = nn.Linear(buffer_size, buffer_dim, bias=False)

    def forward(self, x, q, mean:list, std:list):
        assert len(mean) == 2 and len(std) == 2, "The length of mean index and std index must be 2, but get {} and {}".format(len(mean), len(std))
        assert mean[1] - mean[0] == std[1] - std[0]
        att = self.k(q)
        hard_att = gumbel_softmax(att, temperature=0.1, hard=True)

        di = self.v(hard_att)
        
        di_mean = di[:, mean[0]: mean[1]].unsqueeze(-1).unsqueeze(-1)
        di_std = di[:, std[0]: std[1]].unsqueeze(-1).unsqueeze(-1)

        di_x = adaptive_instance_normalization_meanstd(x, di_mean, di_std)

        return di_x

    def get_di(self):
        buffer = self.v.weight.data.detach()
        print("Get the buffer size: {}".format(buffer.shape))
        return buffer



class diBufferSelfAtt(nn.Module):
    def __init__(self, buffer_dim:int, buffer_size=200):
        super(diBufferSelfAtt, self).__init__()

        self.buffer = nn.Parameter(torch.rand(buffer_size, buffer_dim))

    def forward(self, x, q, mean:list, std:list, getID=False):
        assert len(mean) == 2 and len(std) == 2, "The length of mean index and std index must be 2, but get {} and {}".format(len(mean), len(std))
        assert mean[1] - mean[0] == std[1] - std[0]
        q = F.normalize(q, p=2, dim=1)
        att = torch.mm(q, F.normalize(self.buffer, p=2, dim=1).transpose(0, 1))
        hard_att = gumbel_softmax(att, temperature=0.1, hard=True)

        if getID:
            return hard_att

        di = torch.mm(hard_att, self.buffer)

        return di, hard_att

    def get_di(self):
        buffer = self.buffer.data.detach()
        print("Get the buffer size: {}".format(buffer.shape))
        return buffer



class diBufferSelfAttRealFake(nn.Module):
    def __init__(self, buffer_dim:int, buffer_size=200):
        super(diBufferSelfAttRealFake, self).__init__()

        self.buffer = nn.Parameter(torch.rand(buffer_size, buffer_dim), requires_grad=True)

    def forward(self, q, label, getID=False):
        q = F.normalize(q, p=2, dim=1)
        self.buffer.to("cuda")
        q = q.to("cuda")
        att = torch.mm(q, F.normalize(self.buffer, p=2, dim=1).transpose(0, 1).cuda())

        self_att = torch.mm(F.normalize(self.buffer, p=2, dim=1).cuda(), F.normalize(self.buffer, p=2, dim=1).transpose(0, 1).cuda())

        mask_real = torch.zeros(1, att.shape[1])
        mask_real[:, att.shape[1] // 2 : ] = -1e7
        mask_fake = torch.zeros(1, att.shape[1])
        mask_fake[:, 0 : att.shape[1] // 2] = -1e7

        mask = []

        for i in range(label.shape[0]):
            if label[i] == 0:
                mask.append(mask_real)
            else:
                mask.append(mask_fake)

        mask = torch.cat(mask, dim=0).cuda()
        att_mask = att + mask       

        hard_att = gumbel_softmax(att_mask, temperature=0.1, hard=True)

        if getID:
            return hard_att

        di = torch.mm(hard_att.cuda(), self.buffer.cuda())

        return di, hard_att, att, self_att

    def get_di(self):
        buffer = self.buffer.data.detach()
        print("Get the buffer size: {}".format(buffer.shape))
        return buffer

    
    def input_stype(self, data):
        self.buffer.copy_(data)