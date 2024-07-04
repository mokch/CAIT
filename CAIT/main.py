from train_model import trainer
import os 

# os.environ["CUDA_VISIBLE_DEVICES"] = 'MIG-dc45e153-fb1e-5b2d-8a31-d3fb9494cd80'
data_dir = './datasets'
save_dir = './result'

data_name_list = os.listdir(data_dir)
# data_split_seed_list = [2028, 2029, 2030, 2031, 2032]
# data_split_seed_list = [2023, 2024, 2025, 2026, 2027]
# data_split_seed_list = [2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037]
# data_split_seed_list = [2031, 2032, 2033, 2034, 2035]
# data_split_seed_list = [2024, 2029]
# data_split_seed_list = [2025, 2030]
# data_split_seed_list = [2026, 2031]
# data_split_seed_list = [2027, 2032]
# data_split_seed_list = [2028, 2033]
data_split_seed_list = [2029, 2030, 2031, 2032, 2033]

# data_name_list = data_name_list[4:]
data_name_list = ['seismic.arff']

pass_list = []
# pass_list = ['kddcup.data_10_percent_corrected.zip', 'kddcup.data_10_percent_corrected_rev.zip']

for data_split_seed in data_split_seed_list:
    for data_name in data_name_list:
        if data_name in pass_list:
            pass
        else:

            print(data_name)
            Trainer = trainer(data_name = data_name, data_path = data_dir, data_split_seed = data_split_seed)
            Trainer.set_save_param(save_dir = save_dir, loss_plot = True, save_result = True)
            Trainer.load_dataloader(batch_size = 32)
            if Trainer.check_save():
                pass
            else:
                Trainer.load_dataloader(batch_size = 32)
                Trainer.set_model(model_version = 'v1', depth = 2, embed_dim = 256, device = 'cuda:0')
                Trainer.set_train_param(patience = -1, optimizer = 'adam', epochs = 100, num_corruption = 64, model_loss = 'cos_ratio')
                Trainer.train()
                Trainer.prepare_test()
                Trainer.test() 
