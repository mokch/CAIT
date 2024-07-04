import torch.nn as nn
import torch.nn.functional as F

import torch


device = torch.device('cuda:0')

class InfoNCE_Loss(nn.Module):
    def __init__(self, temperature):
        super(InfoNCE_Loss, self).__init__()
        self.temperature = temperature
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):

        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        N_pos = z_i.shape[0]
        N_neg = z_j.shape[0]
        sim_pos = self.similarity_f(z_i.unsqueeze(1), z_i.unsqueeze(0)) / self.temperature
        sim_neg = self.similarity_f(z_i.unsqueeze(1), z_j.unsqueeze(0)) / self.temperature

        sim_pos = sim_pos.reshape(N_pos*N_pos, -1)
        mask = torch.zeros(sim_pos.shape[0], sim_pos.shape[1], device=device)
        # top_sim = torch.where(sim_pos > 0, sim_pos, mask)
        top_sim = sim_pos
        pos_exp = torch.exp(top_sim)
        neg_exp = torch.exp(sim_neg.reshape(N_pos*N_neg, -1))


        return -torch.log(pos_exp.sum()/(neg_exp.sum()+pos_exp.sum()))
    
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