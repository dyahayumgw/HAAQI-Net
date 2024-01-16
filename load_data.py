import os
import time
import librosa
import random
import numpy as np
import pandas as pd
import collections
from scipy.io import wavfile
from scipy.signal import hamming
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import math
from ast import literal_eval
maxv = np.iinfo(np.int16).max

# get the current working directory
current_working_directory = os.getcwd()

class Dataset_train(Dataset):  
    def __init__(self, filepath):
        self.data_list = filepath
        self.ref = self.data_list['ref'].to_numpy()
        self.data = self.data_list['data'].to_numpy()
        self.mode = "train"
        str2array = lambda x: np.fromstring(x.replace('\n','').replace('[','').replace(']','').replace('  ',' '), sep=' ')
        self.hl = self.data_list['HL'].to_numpy()
        self.HAAQIscore = self.data_list['HAAQI'].astype('float32').to_numpy()
                      
    def __getitem__(self, idx):
        
        beats_data = torch.load('../beats_features' + self.data[idx].replace(".wav", ".pt")).detach()
        beats_data = torch.squeeze(beats_data, dim=0).permute(1, 0)
        
        hl = np.asarray(literal_eval(self.hl[idx]))
        haaqi = self.HAAQIscore[idx]
        
        return beats_data, \
               torch.from_numpy(hl).float(), torch.from_numpy(np.asarray(haaqi)).float()
        
    def __len__(self):
        return len(self.data_list)
    
class Dataset_test(Dataset):  
    def __init__(self, filepath):
        self.data_list = filepath
        self.ref = self.data_list['ref'].to_numpy()
        self.data = self.data_list['data'].to_numpy()
        str2array = lambda x: np.fromstring(x.replace('\n','').replace('[','').replace(']','').replace('  ',' '), sep=' ')
        self.hl = self.data_list['HL'].to_numpy()
        self.HAAQIscore = self.data_list['HAAQI'].astype('float32').to_numpy()
        self.hltype = self.data_list['HLType'].to_numpy()
    
    def __getitem__(self, idx):
        data_name = self.data[idx]
        beats_data = torch.load('../beats_features' + self.data[idx].replace(".wav", ".pt")).detach()

        beats_data = torch.squeeze(beats_data, dim=0).permute(1, 0)
        
        hl = np.asarray(literal_eval(self.hl[idx]))
        haaqi = self.HAAQIscore[idx]
        hltype = self.hltype[idx]
        return data_name, beats_data,\
               torch.from_numpy(hl).float(), torch.from_numpy(np.asarray(haaqi)).float(), hltype 
    
    def __len__(self):
        return len(self.data_list)
    
