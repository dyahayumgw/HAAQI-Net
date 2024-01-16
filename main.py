import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import gc
import yaml
import random
import argparse
import numpy as np
import pandas as pd

import torch
torch.cuda.device_count()
import torch.nn as nn
from torch.utils.data import DataLoader
from getpath import get_trainfile, get_testfile 
from load_data import Dataset_train, Dataset_test
import module 
from trainer import train, test
import pdb
from sklearn.model_selection import train_test_split
import multiprocessing

# Set the working directory to the one containing your file
os.chdir('/data2/user_dyah/HAAQI-Net_BEATs_256_dense_github')

def yaml_config_hook(config_file):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)
    if "defaults" in cfg.keys():
        del cfg["defaults"]
    return cfg

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Define a function to set the 'type' column based on 'noise_type'
def set_type(row, unseen_types):
    return 'unseen' if row['noise_type'] in unseen_types else 'seen'

def main(config_path):   
    # Arguments
    parser = argparse.ArgumentParser(description="Combine_Net")
    config = yaml_config_hook(config_path)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    print(f'save dic:{args.train_checkpoint_dir}')
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setup_seed(args.seed)
    
    if args.Train:           
        df_train = pd.read_csv(args.train_filepath, header=0, sep=';')
        df_train, df_valid = train_test_split(df_train, test_size=0.2, random_state=args.seed)
        print(f'Train length:{len(df_train)}, Valid length:{len(df_valid)}')
        train_data = Dataset_train(df_train)
        valid_data = Dataset_test(df_valid)
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, drop_last=True, 
                                  num_workers=args.num_workers, shuffle=True)
        valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, drop_last=True,
                                  num_workers=args.num_workers, shuffle=True) 
        
        model = getattr(module, args.model)(args.input_size, args.hidden_size, args.num_layers, args.dropout,\
                                            args.linear_output, args.act_fn)       
        # model = model.cuda()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        model.to(args.device)
        
        if args.train_continue == True:
            print(f'Loading the model from training dic:{args.train_checkpoint_dir}')   
            ckpt = torch.load(args.train_checkpoint_dir+f'best_loss.pth')['model']
            model.load_state_dict(ckpt)
        
        # Calculate the number of parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
        print(f"Number of trainable parameters in the model: {num_params}")  
        
        
        # Save the number of parameters to a text file
        with open('parameter_count.txt', 'w') as file:
            file.write(f"Number of trainable parameters in the model: {num_params}\n")   
        
        train(model, train_loader, valid_loader, args) 
        
    else:        
        df = pd.read_csv(args.test_filepath, header=0, sep=';')
        df['noise_type'] = df['data'].apply(lambda x: x.split("/")[2][10:x.split("/")[2].rfind('_')])
        df.loc[df["noise_type"] == "", "noise_type"] = "clean"

        # Apply the function to create the 'type' column
        df['type'] = df.apply(lambda row: set_type(row, args.unseen_types), axis=1)
        df_seen = df[df["type"] == "seen"]
        df_unseen = df[df["type"] == "unseen"] 
        print(f'Seen length:{len(df_seen)}, Unseen length:{len(df_unseen)}')

        model = getattr(module, args.model)(args.input_size,args.hidden_size,args.num_layers,args.dropout,\
                                            args.linear_output,args.act_fn)        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        model.to(args.device)
        print(f'Loading the model from training dic:{args.train_checkpoint_dir}')   
        ckpt = torch.load(args.train_checkpoint_dir+f'best_loss.pth')['model']
        model.load_state_dict(ckpt) 
        
        # seen
        data_seen = Dataset_test(df_seen)
        seen_loader = DataLoader(data_seen, batch_size=1, shuffle=False, num_workers=args.num_workers)
        print("==> Testing for Seen HL...")   
        test(model, seen_loader, 'seen', args)
    
        # unseen
        data_unseen = Dataset_test(df_unseen)
        unseen_loader = DataLoader(data_unseen, batch_size=1, shuffle=False, num_workers=args.num_workers)
        print("==> Testing for Unseen HL...")   
        test(model, unseen_loader, 'unseen', args)

if __name__ == "__main__":    
    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn')
    config_path = "hyper.yaml"
    print(config_path)

    process = multiprocessing.Process(target=main(config_path))
    process.start()
    process.join()
