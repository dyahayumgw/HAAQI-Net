import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
# from util import get_feature, feature_to_wav

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

class BLSTM_frame_sig_att(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, linear_output, act_fn, beats_model):
        super().__init__()
        self.blstm = nn.LSTM(input_size = input_size, 
                             hidden_size = hidden_size,
                             num_layers = num_layers, 
                             dropout = dropout, 
                             bidirectional = True, 
                             batch_first = True)
        # self.transform = get_feature()
        self.dim = 768
        self.beats_model = beats_model
        self.linear0 = nn.Linear(self.dim, linear_output, bias=True)
        self.linear1 = nn.Linear(hidden_size*2, linear_output, bias=True)
        self.act_fn = get_act_fn(act_fn)
        self.dropout = nn.Dropout(p=0.3)
        self.haaqiAtt_layer = nn.MultiheadAttention(linear_output, num_heads=16)
        
        self.haaqiframe_score = nn.Linear(linear_output, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.haaqiaverage_score = nn.AdaptiveAvgPool1d(1)
        
        # weighted sum
        weight_dim = 12
        self.weights = nn.Parameter(torch.ones(weight_dim))
        self.softmax = nn.Softmax(-1)
        layer_norm  = []
        for _ in range(weight_dim):
            layer_norm.append(nn.LayerNorm(self.dim))
        self.layer_norm = nn.Sequential(*layer_norm)
            
    def forward(self, x, hl, layer_norm=True):
        x, xs = self.beats_model.extract_features(x[0])[:2]
        xs = torch.cat(xs,2)
        B, Freq, embed_dim = xs.size()
        lms  = torch.split(xs, self.dim, dim=2)
        all_TE_out = []
        for i,(lm,layer,weight) in enumerate(zip(lms,self.layer_norm,self.softmax(self.weights))):
            if layer_norm:
                lm = layer(lm)
            if i==0:
                out = lm*weight
            else:
                out = out+lm*weight
            all_TE_out.append(out)
        weighted_sum_x = out
        
        x_reduced = self.linear0(weighted_sum_x)
        hl = hl.unsqueeze(1)
        hl_repeat = hl.repeat(1,Freq,1)
        x_concate = torch.cat((x_reduced,hl_repeat), 2)
        
        blstm_out, _ = self.blstm(x_concate)
        out = self.dropout(self.act_fn(self.linear1(blstm_out))).transpose(0,1) 
        att_out, _ = self.haaqiAtt_layer(out,out,out)

        haaqi= att_out.transpose(0,1)
        haaqi= self.haaqiframe_score(haaqi) 
        haaqi= self.sigmoid(haaqi)
        haaqi_fram= haaqi.permute(0,2,1)
        haaqi_avg= self.haaqiaverage_score(haaqi_fram)
        
        return haaqi_fram, haaqi_avg.squeeze(1), blstm_out, att_out, all_TE_out
       
