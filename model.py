import torch
import torch.nn as nn
import numpy as np

    
class MLP(torch.nn.Sequential):

    def __init__(self, input_dim, hidden_dim, n_layers):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_dim))

        super().__init__(*layers)


class Pretext_model_v1(nn.Module):
    def __init__(self, input_dim, emb_dim, depth=6):
        super().__init__()
        self.encoder = MLP(input_dim, emb_dim, depth)
        self.similarity_f = nn.CosineSimilarity(dim=1)
        
    def normalize(self, x):
        return nn.functional.normalize(x, p=2, dim=1)
    
    def cos_sim(self, x, y):
        return self.similarity_f(x, y)
        
    def forward(self, x: torch.Tensor):              
        return self.encoder(x)
