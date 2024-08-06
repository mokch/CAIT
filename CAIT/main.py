from train_model import trainer
import os 


data_dir = './datasets'
save_dir = './result'
data_split_see = 2024
data_name = 'wine.mat'

def main():

    Trainer = trainer(data_name = data_name, data_path = data_dir, data_split_seed = data_split_seed)
    Trainer.set_save_param(save_dir = save_dir, loss_plot = True, save_result = True)
    Trainer.load_dataloader(batch_size = 32)
    Trainer.set_model(model_version = 'v1', depth = 2, embed_dim = 256, device = 'cuda:0')
    Trainer.set_train_param(patience = -1, optimizer = 'adam', epochs = 100, num_corruption = 64, model_loss = 'cos_ratio')
    Trainer.train()
    Trainer.prepare_test()
    Trainer.test() 
    
if __name__ == "__main__":
    main()
