import torch
import torch.nn as nn
import torch.nn.functional as F

class frame_mse(nn.Module):
    def __init__(self):
        super(frame_mse, self).__init__()

    def forward(self, y_pred, y_true): #(B,1,T) (B)
        y_pred = y_pred.squeeze(1) #(B,T)
        B,T = y_pred.size()
        y_true_repeat = y_true.unsqueeze(1).repeat(1,T) #(B,T)
        loss = torch.mean((y_true_repeat - y_pred.detach()) ** 2)
        return loss
    
