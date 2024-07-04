import numpy as np
import torch

def make_fake_samples_2(input_data, num_samples, max_list, min_list, percentage=1.):

    norm_data = torch.clone(input_data)
    fake_data = torch.empty(num_samples, input_data.shape[1], dtype=torch.float)

    numerical_variables = list(range(input_data.shape[1]))
    selected_samples = np.random.choice(norm_data.shape[0],num_samples)

    for i in range(num_samples):
        fake_data[i] = norm_data[selected_samples[i]]
        for j in np.random.permutation(numerical_variables)[:len(max_list)//4]:
            fake_data[i][j] = max_list[j]*percentage
        for j in np.random.permutation(numerical_variables)[len(max_list)//4:len(max_list)//2]:
            fake_data[i][j] = min_list[j]*percentage
    return fake_data

def make_fake_samples(input_data, num_samples, num_variables, max_list, percentage=1.):

    norm_data = torch.clone(input_data)
    fake_data = torch.empty(num_samples, input_data.shape[1], dtype=torch.float)

    numerical_variables = list(range(50,92,1))
    selected_samples = np.random.choice(norm_data.shape[0],num_samples)

    for i in range(num_samples):
        fake_data[i] = norm_data[selected_samples[i]]
        for j in np.random.permutation(numerical_variables)[:num_variables]:
            fake_data[i][j] = max_list[j]*percentage

    return fake_data

def make_fake_samples_min(input_data, num_samples, max_list, min_list, percentage=1.):

    norm_data = torch.clone(input_data)
    fake_data = torch.empty(num_samples, input_data.shape[1], dtype=torch.float)

    numerical_variables = list(range(input_data.shape[1]))
    selected_samples = np.random.choice(norm_data.shape[0],num_samples)

    for i in range(num_samples):
        fake_data[i] = norm_data[selected_samples[i]]
        for j in np.random.permutation(numerical_variables)[:len(max_list)//4]:
            fake_data[i][j] = min_list[j]*percentage
        for j in np.random.permutation(numerical_variables)[len(max_list)//4:len(max_list)//2]:
            fake_data[i][j] = min_list[j]*percentage
    return fake_data


def make_fake_samples_max(input_data, num_samples, max_list, min_list, percentage=1.):

    norm_data = torch.clone(input_data)
    fake_data = torch.empty(num_samples, input_data.shape[1], dtype=torch.float)

    numerical_variables = list(range(input_data.shape[1]))
    selected_samples = np.random.choice(norm_data.shape[0],num_samples)

    for i in range(num_samples):
        fake_data[i] = norm_data[selected_samples[i]]
        for j in np.random.permutation(numerical_variables)[:len(max_list)//4]:
            fake_data[i][j] = max_list[j]*percentage
        for j in np.random.permutation(numerical_variables)[len(max_list)//4:len(max_list)//2]:
            fake_data[i][j] = max_list[j]*percentage
    return fake_data

def make_fake_samples_mean(input_data, num_samples, max_list, min_list, percentage=1.):

    norm_data = torch.clone(input_data)
    fake_data = torch.empty(num_samples, input_data.shape[1], dtype=torch.float)

    numerical_variables = list(range(input_data.shape[1]))
    selected_samples = np.random.choice(norm_data.shape[0],num_samples)

    for i in range(num_samples):
        fake_data[i] = norm_data[selected_samples[i]]
        for j in np.random.permutation(numerical_variables)[:len(max_list)//4]:
            fake_data[i][j] = 0.0001
        for j in np.random.permutation(numerical_variables)[len(max_list)//4:len(max_list)//2]:
            fake_data[i][j] = 0.0001
    return fake_data


def random_corruption_by_rate(data, corruption_rate = 0.5):
    batch_size, feature_size = data.shape
    device = data.device
    mask_index = torch.randperm(feature_size, device=device)
    for i in range(batch_size):
        for j in range(int(feature_size*corruption_rate)):
            data[i,mask_index[j]] = torch.randn(device=device)

    return data

def random_corruption_by_num(data, corruption_num = 2):
    batch_size, feature_size = data.shape
    device = data.device
    mask_index = torch.randperm(feature_size, device=device)
    for i in range(batch_size):
        for j in range(corruption_num):
            data[i,mask_index[j]] = torch.randn(device=device)

    return data

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

def random_corruption_by_variable(data, num, var_num, mask_num = 1):
    batch_size, feature_size = data.shape
    device = data.device

    new_data = data.clone().detach()
    new_data = new_data.repeat(num,1)

    y = torch.zeros(batch_size*num, device=device)

    for i in range(batch_size*num):
        y[i] = torch.tensor(1-mask_num/feature_size)
        new_data[i,var_num] = torch.rand(1, device=device)  

    return new_data, y

def random_corruption_by_mask_num(data, num, var_num, mask_num = 1):
    batch_size, feature_size = data.shape
    device = data.device

    new_data = data.clone().detach()
    new_data = new_data.repeat(num,1)

    y = torch.zeros(batch_size*num, device=device)

    for i in range(batch_size*num):
        y[i] = torch.tensor(1-mask_num/feature_size)
        mask_index = torch.randperm(feature_size, device=device)
        for j in range(mask_num):
            # new_data[i,mask_index[j]] = torch.randn(1, device=device)  
            new_data[i,mask_index[j]] = torch.rand(1, device=device)  

    return new_data, y