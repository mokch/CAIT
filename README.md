# CAIT
This repository contains the original PyTorch implementation of the paper 'Corruption-Based Anomaly Detection and Interpretation in Tabular Data'.

## A. Dataset
1. Download from the following link: [Outlier Detection DataSets (ODDS)](https://odds.cs.stonybrook.edu/)
2. Place the each dataset file under `./CAIT/datasets/`.
3. Run the following script from the working directory:
```
python main.py --data_name wine.mat --num_corruption 64 --data_split_seed 2024 
```
## B. Available Parameters
--data_dir: Directory for datasets (default: ./datasets)
--save_dir: Directory for saving results (default: ./result)
--data_name: Name of the data file (default: wine.mat)
--data_split_seed: Seed for data splitting (default: 2024)
--batch_size: Batch size for training (default: 32)
--embed_dim: Embedding dimension for the model (default: 256)
--device: Device to use for training (e.g., cuda or cpu, default: cuda)
--epochs: Number of epochs for training (default: 100)
--num_corruption: Number of corruptions for training (default: 64)
