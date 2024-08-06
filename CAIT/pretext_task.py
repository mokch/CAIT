import numpy as np
import torch

def random_corruption(data, num):
    batch_size, feature_size = data.shape
    device = data.device

    new_data = data.clone().detach()
    new_data = new_data.repeat(num,1)

    y = torch.zeros(batch_size*num, device=device)

    for i in range(batch_size*num):
        mask_num = torch.randint(1, feature_size+1, (1,1), device=device).item()
        y[i] = torch.tensor(1-mask_num/feature_size)
        mask_index = torch.randperm(feature_size, device=device)
        for j in range(mask_num):
            # new_data[i,mask_index[j]] = torch.randn(1, device=device)  
            new_data[i,mask_index[j]] = torch.rand(1, device=device)  

    return new_data, y
