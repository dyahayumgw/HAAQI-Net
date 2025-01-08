import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb 

def get_act_fn(act_fn):
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.3)
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError('Invalid argument for act_fn') 

class haaqi_net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, linear_output, act_fn):
        super().__init__()
        self.blstm = nn.LSTM(input_size = input_size, 
                             hidden_size = hidden_size,
                             num_layers = num_layers, 
                             dropout = dropout, 
                             bidirectional = True, 
                             batch_first = True)
        self.beats_size = 768
        self.linear0 = nn.Linear(self.beats_size, linear_output, bias=True)
        self.linear1 = nn.Linear(hidden_size*2, linear_output, bias=True)
        self.act_fn = get_act_fn(act_fn)
        self.dropout = nn.Dropout(p=0.3)
        self.haaqiAtt_layer = nn.MultiheadAttention(linear_output, num_heads=16)
        
        self.haaqiframe_score = nn.Linear(linear_output, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.haaqiaverage_score = nn.AdaptiveAvgPool1d(1)
            
    def forward(self, x, hl):
        B, Freq, T = x.size()
        x = x.permute(0,2,1)
        x_reduced = self.linear0(x)
        hl = hl.unsqueeze(1)
        hl_repeat = hl.repeat(1,T,1)
        x_concate = torch.cat((x_reduced,hl_repeat), 2)
        
        out, _ = self.blstm(x_concate)
        out = self.dropout(self.act_fn(self.linear1(out))).transpose(0,1) 
        haaqi, _ = self.haaqiAtt_layer(out,out,out)

        haaqi= haaqi.transpose(0,1)
        haaqi= self.haaqiframe_score(haaqi) 
        haaqi= self.sigmoid(haaqi)
        haaqi_fram= haaqi.permute(0,2,1)
        haaqi_avg= self.haaqiaverage_score(haaqi_fram)
        
        return haaqi_fram, haaqi_avg.squeeze(1)
