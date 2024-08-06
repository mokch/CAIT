import argparse
from train_model import trainer
import os 


def main(args):
    Trainer = trainer(data_name=args.data_name, data_path=args.data_dir, data_split_seed=args.data_split_seed)
    Trainer.set_save_param(save_dir=args.save_dir, loss_plot=True, save_result=True)
    Trainer.load_dataloader(batch_size = args.batch_size)
    Trainer.set_model(model_version = 'v1', depth = 2, embed_dim = 256, device = args.device)
    Trainer.set_train_param(patience = -1, optimizer = 'adam', epochs = args.epochs, num_corruption = args.num_corruption, model_loss = 'cos_ratio')
    Trainer.train()
    Trainer.prepare_test()
    Trainer.test() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specified parameters")
    parser.add_argument('--data_name', type=str, default='wine.mat', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./datasets', help='Directory for datasets')
    parser.add_argument('--save_dir', type=str, default='./result', help='Directory for saving results')
    parser.add_argument('--data_split_seed', type=int, default=2024, help='Seed for data splitting')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cuda:0 or cpu)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--num_corruption', type=int, default=64, help='Number of corruptions for training')

    args = parser.parse_args()
    main(args)



