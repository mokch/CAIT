import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
from scipy import io
from sklearn.model_selection import train_test_split
import zipfile
import mat73
from collections import Counter

class ODDSLoader(object):
    def __init__(self, data_path, data_split_seed, test_size, mode="train"):

        self.mode = mode
        self.scaler = MinMaxScaler()
        normal, abnormal = reader(data_path)

        train_X_, test_X = train_test_split(normal, test_size=test_size, random_state=data_split_seed)
        train_X, valid_X = train_test_split(train_X_, test_size=0.1, random_state=data_split_seed)
        test_X = np.concatenate((test_X, abnormal), axis = 0)
        test_y = np.concatenate((np.zeros(test_X.shape[0]-abnormal.shape[0]), np.ones(abnormal.shape[0])))

        self.scaler.fit(train_X_)
        self.train = self.scaler.transform(train_X_)
        self.valid = self.scaler.transform(valid_X)
        self.test = self.scaler.transform(test_X)
        self.test_labels = test_y
        
        if mode == 'train':
            print("test:", self.test.shape)
            print("train:", self.train.shape)
            print("valid:", self.valid.shape)


    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif self.mode == 'valid':
            return self.valid.shape[0]
        elif self.mode == 'test':
            return self.test.shape[0]
        else:
            return self.test.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.zeros(self.train.shape[0])[index]
        elif self.mode == 'valid':
            return np.float32(self.valid[index]), np.zeros(self.valid.shape[0])[index]
        elif self.mode == 'test':
            return np.float32(self.test[index]), np.float32(self.test_labels[index])
        else:
            return 0        
        
    def feature_dim(self):
        return self.train.shape[1]
    



def get_loader(data_path, batch_size, data_split_seed, test_size, mode):

    dataset = ODDSLoader(data_path, data_split_seed, test_size, mode=mode)
    feature_dim = dataset.feature_dim()

    if mode == 'test':
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
        return dataloader, feature_dim
    
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        return dataloader, feature_dim
    
def ICL_ODDSLoader(data_path, data_split_seed, test_size):
    
    normal, abnormal = reader(data_path)
    train_X_, test_X = train_test_split(normal, test_size=test_size, random_state=data_split_seed)
    test_X = np.concatenate((test_X, abnormal), axis = 0)
    test_y = np.concatenate((np.zeros(test_X.shape[0]-abnormal.shape[0]), np.ones(abnormal.shape[0])))
    
    std_scaler = StandardScaler()
    std_scaler.fit(train_X_)
    train_X_ = std_scaler.transform(train_X_)
    test_X = std_scaler.transform(test_X)


    return train_X_, test_X, test_y
    
    
def reader(data_path):
    
    if data_path.split('/')[-1] == "ecoli.data":
        data = pd.read_csv(data_path, header=None, sep='\s+')
        data = data.iloc[:, 1:]
        abnormal = np.array(data[(data.iloc[:, 7] == 'omL') | (data.iloc[:, 7] == 'imL') | (data.iloc[:, 7] == 'imS')])
        normal = np.array(data[(data.iloc[:, 7] == 'cp') | (data.iloc[:, 7] == 'im') | (data.iloc[:, 7] == 'pp') | (data.iloc[:, 7] == 'imU') | (data.iloc[:, 7] == 'om')])
        abnormal = abnormal[:,:-1]
        normal = normal[:,:-1]
        
    elif data_path.split('/')[-1] == "abalone.data":
        data = pd.read_csv(data_path, header=None, sep=',')
        data = data.rename(columns={8: 'y'})
        X = data.iloc[:,1:8]
        X_cat = data.iloc[:,0]
        X_cat = pd.get_dummies(X_cat)
        X = pd.concat([X_cat,X], axis=1)
        y = data.iloc[:,8]
        y.replace([8, 9, 10], 0, inplace=True)
        y.replace([3, 21], 1, inplace=True)

        normal = np.array(X[y == 0])
        abnormal = np.array(X[y == 1])
        
        
    elif data_path.split('/')[-1] == "kddcup.data_10_percent_corrected.zip":
        zf = zipfile.ZipFile(data_path)       
        kdd_loader = pd.read_csv(zf.open('kddcup.data_10_percent_corrected'), delimiter=',')
        entire_set = np.array(kdd_loader)
        data = pd.DataFrame(entire_set)
        
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        new_data = pd.DataFrame()
        
        for i in range(X.shape[1]):
            aa = len(Counter(X.iloc[:,i]))
            if i == 2:
                new_data = pd.concat([new_data, pd.get_dummies(X.iloc[:,i])], axis=1)
            elif 1<aa and aa<20:
                new_data = pd.concat([new_data, pd.get_dummies(X.iloc[:,i])], axis=1)
            elif aa == 1:
                pass
            else:
                new_data = pd.concat([new_data,pd.DataFrame(X.iloc[:,i])], axis=1)
                
        X = np.array(new_data)
        y = np.array(y)
        normal = X[y=='normal.']
        abnormal = X[y!='normal.']        
        
   
    elif data_path.split('/')[-1] == "kddcup.data_10_percent_corrected_rev.zip":
        zf = zipfile.ZipFile(data_path)
        kdd_loader = pd.read_csv(zf.open('kddcup.data_10_percent_corrected'), delimiter=',')
        entire_set = np.array(kdd_loader)
        data = pd.DataFrame(entire_set)
        
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        new_data = pd.DataFrame()
        
        for i in range(X.shape[1]):
            aa = len(Counter(X.iloc[:,i]))
            if i == 2:
                new_data = pd.concat([new_data, pd.get_dummies(X.iloc[:,i])], axis=1)
            elif 1<aa and aa<20:
                new_data = pd.concat([new_data, pd.get_dummies(X.iloc[:,i])], axis=1)
            elif aa == 1:
                pass
            else:
                new_data = pd.concat([new_data,pd.DataFrame(X.iloc[:,i])], axis=1)
                
        X = np.array(new_data)
        y = np.array(y)
        normal = X[y=='normal.']
        abnormal = X[y!='normal.']  

    elif data_path.split('/')[-1] == "mulcross.arff":
        data, _ = io.arff.loadarff(data_path)
        data = pd.DataFrame(data)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        normal = X[y==b'Normal']
        abnormal = X[y==b'Anomaly']
        
    elif data_path.split('/')[-1] == "seismic.arff":

        data, _ = io.arff.loadarff(data_path)
       
        data = pd.DataFrame(data)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        new_data = pd.DataFrame()
        
        for i in range(X.shape[1]):
            aa = len(Counter(X.iloc[:,i]))
            if 1<aa and aa<20:
                new_data = pd.concat([new_data, pd.get_dummies(X.iloc[:,i])], axis=1)
            elif aa == 1:
                pass
            else:
                new_data = pd.concat([new_data,pd.DataFrame(X.iloc[:,i])], axis=1)
                
        X = np.array(new_data)
        y = np.array(y)
        normal = X[y==b'0']
        abnormal = X[y==b'1']        
        
    elif data_path.split('/')[-1] == "smtp.mat" or data_path.split('/')[-1] == "http.mat":
        data = mat73.loadmat(os.path.join(data_path))

        X = data['X']
        y = np.array(data['y']).reshape(-1)
        normal = X[y==0]
        abnormal = X[y==1]
        
    elif data_path.split('./')[-1] == "lympho.mat" or data_path.split('./')[-1] == "arrhythmia.mat" or data_path.split('./')[-1] == "breastw.mat":
        
        data = io.loadmat(os.path.join(data_path))
        new_data = pd.DataFrame()
        
        for i in range(data['X'].shape[1]):
            aa = len(Counter(data['X'][:,i]))
            if 1<aa and aa<20:
                new_data = pd.concat([new_data, pd.get_dummies(data['X'][:,i])], axis=1)
            elif aa == 1:
                pass
            else:
                new_data = pd.concat([new_data,pd.DataFrame(data['X'][:,i])], axis=1)
                
        X = np.array(new_data)
        y = np.array(data['y']).reshape(-1)
        normal = X[y==0]
        abnormal = X[y==1]          
        
    else:
        
        data = io.loadmat(os.path.join(data_path))

        X = data['X']
        y = np.array(data['y']).reshape(-1)
        normal = X[y==0]
        abnormal = X[y==1]

    return normal.astype(np.float64), abnormal.astype(np.float64)
