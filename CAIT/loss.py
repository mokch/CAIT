import torch.nn as nn
import torch.nn.functional as F

import torch


device = torch.device('cuda:0')

class Ratio_Loss(nn.Module):
    def __init__(self):
        super(Ratio_Loss, self).__init__()
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.mse = nn.MSELoss()

    def forward(self, pos_x, neg_x, neg_y):

        pos_x = F.normalize(pos_x, p=2, dim=1)
        neg_x = F.normalize(neg_x, p=2, dim=1)
        device = pos_x.device

        N_pos = pos_x.shape[0]
        N_neg = neg_x.shape[0]
        pos_label = torch.ones((N_pos, N_pos), device=device)
        neg_y = neg_y*2
        neg_y = neg_y-1
        neg_label = neg_y.to(device).repeat(N_pos,1)

        sim_pos = self.similarity_f(pos_x.unsqueeze(1), pos_x.unsqueeze(0))
        sim_neg = self.similarity_f(pos_x.unsqueeze(1), neg_x.unsqueeze(0))
        
        loss = self.mse(sim_pos, pos_label) + self.mse(sim_neg, neg_label)

        return loss
