from tokenize import Double
import torch
import numpy as np
import os
from pretext_task import random_corruption
import matplotlib.pyplot as plt
from loss import Ratio_Loss
import pandas as pd
#input = (batch, variables)
#mask_matrix = (num_masks, variables)
from sklearn.metrics import classification_report, roc_auc_score
from model import Pretext_model_v1
import glob
from dataloader import get_loader
import collections

class trainer(object):

    def __init__(self, data_name, data_path = './datasets', data_split_seed = 2023):
        self.data_name = data_name
        self.data_path = data_path
        self.data_split_seed = data_split_seed

    def set_save_param(self, save_dir, save_model = True, loss_plot = True, save_result = True):
        self.save_dir = os.path.join(save_dir, str(self.data_split_seed))
        os.makedirs(save_dir, exist_ok=True)

        self.save_model = save_model
        self.loss_plot = loss_plot
        self.save_result = save_result

    def load_dataloader(self, batch_size = 32):
        data_dir = os.path.join(self.data_path, self.data_name)
        self.train_dataloader, self.feature_dim = get_loader(data_dir, batch_size, self.data_split_seed, test_size = 0.5, mode = 'train')
        self.valid_dataloader, feature_dim = get_loader(data_dir, batch_size, self.data_split_seed, test_size = 0.5, mode = 'valid')
        self.test_dataloader, feature_dim = get_loader(data_dir, batch_size, self.data_split_seed, test_size = 0.5, mode = 'test')

    def set_model(self, model_version = 'v1', depth = 2, embed_dim = 512, device = 'cuda'):
        self.model_version = model_version
        self.depth = depth
        self.embed_dim = embed_dim
        self.device = device

        if self.model_version == 'v1':
            self.model = Pretext_model_v1(self.feature_dim, self.embed_dim, self.depth).to(self.device)

    def set_train_param(self, patience = 10, optimizer = 'adam', epochs = 1000, num_corruption = 128, model_loss = 'cos_ratio'):
        self.patience = patience
        self.num_corruption = num_corruption
        self.epochs = epochs
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        if model_loss == 'cos_ratio':
            self.criterion = Ratio_Loss()


    def _save_model(self):
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'{self.data_name}_lastmodel.pth'))

    def _load_model(self):
        if self.model_version == 'v1':
            self.model = Pretext_model_v1(self.feature_dim, self.embed_dim, self.depth).to(self.device)

        list_of_files = glob.glob(os.path.join(self.save_dir, '*.pth'))
        latest_file = max(list_of_files, key=os.path.getmtime)
        self.model.load_state_dict(torch.load(latest_file))


    def check_save(self):
        acc_result_save_name = f'acc_result_{self.data_name}.csv'
        return os.path.exists(os.path.join(self.save_dir, acc_result_save_name))
    def train(self):

        model_hist = collections.namedtuple('Model', 'epoch loss val_loss val_abnorm_loss')
        model_loss = model_hist(epoch=[], loss=[], val_loss=[], val_abnorm_loss=[])        
        base_loss = 100000
        count = 0 

        for epoch in range(self.epochs):

            losses = []

            try:
                c = model_loss.epoch[-1]
            except:
                c = 0

            self.model.train()
            for batch in self.train_dataloader:

                data, _ = batch
                self.optimizer.zero_grad()
                corrupted_data, y = random_corruption(data, self.num_corruption)
                xt = data
                pos = self.model(xt.to(self.device))
                neg = self.model(corrupted_data.to(self.device))
                lt = self.criterion(pos, neg, y.to(self.device))
                losses.append(lt.item())
                lt.backward()
                self.optimizer.step()

            model_loss.epoch.append(c + epoch)
            model_loss.loss.append(lt.item()/len(self.train_dataloader))
            print(f'Epoch: {epoch}   Train_Loss: {np.mean(losses):.4f} ')


            best_epoch = epoch


        
        if self.save_model:
            self._save_model()
            print("save model")

        if self.loss_plot:
            x = np.linspace(0, epoch, epoch+1)
            plt.cla()
            # plt.plot(x, model_loss.loss, label="loss")
            plt.plot(x, model_loss.loss, label="train_norm_loss")
            # plt.plot(x, model_loss.val_loss, label="val_norm_loss")
            # plt.plot(x, model_loss.val_abnorm_loss, label="val_abnorm_loss")
            plt.legend()
            plt.savefig(os.path.join(self.save_dir, f'Loss_{self.data_name}_{best_epoch}.png'))
            plt.cla()
        
    def prepare_test(self):
        self.model.eval()
        num_batch = 0
        sum_reps = torch.zeros(self.embed_dim).to(self.device)
        with torch.no_grad():
            num_batch = 0
            for batch in self.train_dataloader:
                data, label = batch
                pos = self.model(data.to(self.device))
                sum_reps += pos.mean(0)
                num_batch += 1

        self.center = sum_reps/num_batch

    def test(self):

        if self.save_model:
            self._load_model()
            print("load model")

        score = []
        y_label = []
        self.model.eval()
        with torch.no_grad():

            for batch in self.test_dataloader:
                data, label = batch
                pos = self.model(data.to(self.device))
                pos = self.model.normalize(pos)
                score.append(-self.model.cos_sim(pos, self.center.unsqueeze(0)).item())
                y_label.append(label.item())

        result_df = pd.DataFrame()
        result_df['pred'] = score
        result_df['real'] = np.array(y_label)
        pred = []
        result_array = np.zeros((1, 4))
        col_names = ['precision','recall','f1-score','AUROC']
        acc_result = pd.DataFrame()
        ratio = sum(y_label)/len(y_label)

        threshold = np.percentile(score, (1-ratio)*100)
        for i in range(len(score)):            

            if score[i] > threshold:
                pred.append(1)
            else:
                pred.append(0)

        df = pd.DataFrame(classification_report(y_label, pred, output_dict=True)).transpose()
        result_array[0,0] = df.loc['macro avg', 'precision']
        result_array[0,1] = df.loc['macro avg', 'recall']
        result_array[0,2] = df.loc['macro avg', 'f1-score']
        result_array[0,3] = roc_auc_score(y_label, score, average = 'macro')
        acc_result = pd.DataFrame(result_array, columns = col_names)
            
        if self.save_result:
            
            result_save_name = f'result_{self.data_name}.csv'
            result_df.to_csv(os.path.join(self.save_dir, result_save_name), index=False)

            acc_result_save_name = f'acc_result_{self.data_name}.csv'
            acc_result.to_csv(os.path.join(self.save_dir, acc_result_save_name), index=False)
