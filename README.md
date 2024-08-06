# CAIT
This repository contains the original PyTorch implementation of the paper 'Corruption-Based Anomaly Detection and Interpretation in Tabular Data'.

## A. Dataset
1. Download from the following link: [Outlier Detection DataSets (ODDS)](https://odds.cs.stonybrook.edu/)
2. Place the each dataset file under `./CAIT/datasets/`.
3. Run the following script from the working directory:
```
python main.py --data_name wine.mat --data_split_seed 2024 --num_corruption 64
```
