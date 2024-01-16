import os
import gc
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch_stft import STFT
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from sklearn.metrics import mean_squared_error
import scipy.stats
import module 
import losses
import time

maxv = np.iinfo(np.int16).max

def train(model, train_loader, valid_loader, args):       
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.train_lr)
    frame = getattr(losses, f'{args.frameloss}')().cuda()
    average = getattr(nn, f'{args.loss}')().cuda() 
    print(optimizer)
    print(average)
    
    if not os.path.exists(os.path.dirname(args.train_summaries_dir)):
        os.makedirs(os.path.dirname(args.train_summaries_dir))
    logger_path = args.train_checkpoint_dir + f'loss.log'    
    with open(logger_path, 'w') as logger_file:
        logger_file.write('epoch,train_loss,HAAQI,valid_loss,valid_HAAQI,Pearson_cc\n')    
   
    args.writer = SummaryWriter(args.train_summaries_dir)
    best_valid_loss = np.inf
    beast_valid_lcc = -10
    patience_counter = 0
    
    total_steps = int(len(train_loader))
    for epoch in range(1,args.train_epoch+1):
        tr_loss, tr_haaqi= 0, 0
        tr_haaqi_fram, tr_haaqi_avg= 0,0
        model.train()      
         
        print('Epoch: ' + str(epoch)+'/'+str(args.train_epoch))       
        pbar = tqdm(total=total_steps)
        pbar.n = 0 # move process bar back to zero
        for step, (Sxx_data, hl, HAAQIscore) in enumerate(train_loader):
            Sxx_data = Sxx_data.cuda(non_blocking=True)
            hl = hl.cuda(non_blocking=True)
            HAAQIscore = HAAQIscore.cuda(non_blocking=True)
            optimizer.zero_grad()
            
            haaqi_fram, haaqi_avg = model(Sxx_data, hl)
            haaqi_loss_fram = frame(haaqi_fram, HAAQIscore)
            haaqi_loss_avg = average(haaqi_avg.squeeze(1), HAAQIscore)
            haaqi_loss = haaqi_loss_fram+haaqi_loss_avg
            loss = haaqi_loss
            loss.backward()
            optimizer.step()
            
            tr_haaqi_fram += haaqi_loss_fram.item()
            tr_haaqi_avg += haaqi_loss_avg.item()
            tr_haaqi+=(haaqi_loss_fram.item()+haaqi_loss_avg.item())
            tr_loss+=(haaqi_loss_fram.item()+haaqi_loss_avg.item())
            pbar.update(1)

        pbar.close()  
        epoch_train_loss = tr_loss/len(train_loader) 
        epoch_haaqi = tr_haaqi/len(train_loader)
        epoch_haaqi_fram, epoch_haaqi_avg = tr_haaqi_fram/len(train_loader),tr_haaqi_avg/len(train_loader)
        print(f'train:{len(train_loader)}')
        
        epoch_valid_loss, epoch_valid_haaqi, epoch_valid_haaqi_fram, epoch_valid_haaqi_avg, Pearson_cc_haaqi = evaluate(model, valid_loader, epoch, args)
        print(f'Epoch:{epoch}')
        print(f'Train loss:{epoch_train_loss:.5f},HAAQI:{epoch_haaqi:.5f},{epoch_haaqi_fram:.5f},{epoch_haaqi_avg:.5f}')
        print(f'Validation loss:{epoch_valid_loss},HAAQI:{epoch_valid_haaqi:.5f},{epoch_valid_haaqi_fram:.5f},{epoch_valid_haaqi_avg:.5f},Pearson_cc:{Pearson_cc_haaqi}')
        
        with open(logger_path, 'a') as logger_file:
            logger_file.write(f'{epoch:03d},{epoch_train_loss:4.6e},{epoch_haaqi:1.6e},{epoch_valid_loss:4.6e},{epoch_valid_haaqi:1.6e},[{Pearson_cc_haaqi:3.6e}]\n')
        
        args.writer.add_scalars(f'{args.loss}', {'train': epoch_train_loss}, epoch)
        args.writer.add_scalars(f'{args.loss}', {'valid': epoch_valid_loss}, epoch)
        args.writer.add_scalars('train_loss', {'train': epoch_haaqi}, epoch)
        args.writer.add_scalars('train_loss', {'valid': epoch_valid_haaqi}, epoch)
        args.writer.add_scalars('frame_loss', {'train': epoch_haaqi_fram}, epoch)
        args.writer.add_scalars('frame_loss', {'valid': epoch_valid_haaqi_fram}, epoch)
        args.writer.add_scalars('avg_loss', {'train': epoch_haaqi_avg}, epoch)
        args.writer.add_scalars('avg_loss', {'valid': epoch_valid_haaqi_avg}, epoch)
        args.writer.add_scalars('Pearson_cc', {'haaqi':Pearson_cc_haaqi}, epoch)
        
        if epoch_valid_loss <= best_valid_loss:
            patience_counter = 0
            best_valid_loss = epoch_valid_loss
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }            
            torch.save(state, args.train_checkpoint_dir+f'best_loss.pth')
        else:
            patience_counter += 1
            if patience_counter == args.train_patience:
                print(f'Early stopping... no improvement at epoch:{epoch} after {args.train_patience} epochs')
                break
                  
    epoch_train_loss, epoch_haaqi = None, None
    torch.cuda.empty_cache()
    gc.collect()  


def evaluate(model, valid_loader, epoch, args): 
    frame = getattr(losses, f'{args.frameloss}')().cuda()
    average = getattr(nn, f'{args.loss}')().cuda() 
    print(frame, average)
    model.eval()
    
    epoch_valid_loss, epoch_valid_haaqi= None, None
    epoch_valid_haaqi_fram, epoch_valid_haaqi_avg = None, None
    total_steps = int(len(valid_loader))
    print(f'valid:{len(valid_loader)}')
    
    output_path = args.train_checkpoint_dir + 'validresult.csv'
    output_file = open(output_path, 'w')
    print('data,TrueHAAQI,PredictHAAQI,HL,HLType', file=output_file)
    
    with torch.no_grad(): 
        valid_loss, valid_haaqi = 0, 0
        valid_haaqi_fram, valid_haaqi_avg = 0,0
        for step, (name, Sxx_data, hl, HAAQIscore, hltype) in enumerate(valid_loader):
            Sxx_data = Sxx_data.cuda(non_blocking=True)
            hl = hl.cuda(non_blocking=True)
            HAAQIscore = HAAQIscore.cuda(non_blocking=True)

            haaqi_fram, haaqi_avg = model(Sxx_data, hl)
            haaqi_loss_fram = frame(haaqi_fram, HAAQIscore)
            haaqi_loss_avg = average(haaqi_avg.squeeze(1), HAAQIscore)
            
            valid_haaqi_fram += haaqi_loss_fram.item()
            valid_haaqi_avg += haaqi_loss_avg.item()
            valid_haaqi+=(haaqi_loss_fram.item()+haaqi_loss_avg.item())
            valid_loss+=(haaqi_loss_fram.item()+haaqi_loss_avg.item())
         
            haaqi_score, predict_haaqi_score= HAAQIscore.cpu().numpy(), haaqi_avg.squeeze(1).cpu().numpy()
            
            for i,j,k,l,m, in zip(name,haaqi_score,predict_haaqi_score,hltype,hl.cpu().numpy()):
                print(i,j,k,l,m, sep=',',file=output_file) 
        
        output_file.close()
        epoch_valid_loss = valid_loss/len(valid_loader) 
        epoch_valid_haaqi = valid_haaqi/len(valid_loader)   
        epoch_valid_haaqi_fram =valid_haaqi_fram/len(valid_loader)
        epoch_valid_haaqi_avg = valid_haaqi_avg/len(valid_loader)
        
        # calculate Pearson_cc and draw figure
        df = pd.read_csv(output_path)
        # SRCC is the Spearman Rank Correlation Coefficent
        # LCC is the normal Linear Correlation Coefficient
        haaqi, predict_haaqi = df['TrueHAAQI'].astype('float32').to_numpy(), df['PredictHAAQI'].astype('float32').to_numpy()
        
        srcc1, pvalue = scipy.stats.spearmanr(haaqi,predict_haaqi)
        Pearson_cc1, p_value = scipy.stats.pearsonr(haaqi, predict_haaqi)
        mse1 = mean_squared_error(haaqi, predict_haaqi)
        
        plt.clf() # clear the fig before we draw
        fig, axs = plt.subplots(1, 2, figsize=(8,4))
        axs[0].plot(haaqi,predict_haaqi, 'o', ms=5)
        axs[0].tick_params(labelsize=8)
        axs[0].set_xlabel("True HAAQI")
        axs[0].set_ylabel("Predicted HAAQI")
        axs[0].set_title(f'LCC:{Pearson_cc1:5f},SRCC:{srcc1:5f}')
        
        Pearson_cc = Pearson_cc1
        plt.tight_layout()
        plt.savefig(f'{args.train_checkpoint_dir}scatter.png')
    torch.cuda.empty_cache()
    gc.collect()  
    
    return epoch_valid_loss,epoch_valid_haaqi,epoch_valid_haaqi_fram,epoch_valid_haaqi_avg,Pearson_cc1

def test(model, test_loader, mode, args): 
    frame = getattr(losses, f'{args.frameloss}')().cuda()
    average = getattr(nn, f'{args.loss}')().cuda() 
    print(frame, average)
    model.eval()
    
    if not os.path.exists(f'{args.result_dir}'):
        os.makedirs(f'{args.result_dir}')
    
    output_path = args.result_dir + f'result_{mode}.csv'
    output_file = open(output_path, 'w')
    print('data,TrueHAAQI,PredictHAAQI,HL,HLType', file=output_file)
    
    test_total_loss, test_total_haaqi = None, None
    total_steps = int(len(test_loader))
    print(len(test_loader))

    start_time = time.time()
    
    with torch.no_grad():          
        test_loss, test_haaqi= 0, 0
        for step, (name, Sxx_data, hl, HAAQIscore, hltype) in enumerate(tqdm(test_loader)):
            Sxx_data = Sxx_data.cuda(non_blocking=True)
            hl = hl.cuda(non_blocking=True)
            HAAQIscore = HAAQIscore.cuda(non_blocking=True)
            haaqi_fram, haaqi_avg= model(Sxx_data, hl)
            haaqi_loss_fram = frame(haaqi_fram, HAAQIscore)
            haaqi_loss_avg = average(haaqi_avg.squeeze(1), HAAQIscore)
            haaqi_loss = haaqi_loss_fram+haaqi_loss_avg
            test_haaqi += (haaqi_loss_fram.item() + haaqi_loss_avg.item()) 
            test_loss += (test_haaqi)
            
            haaqi_score, predict_haaqi_score= HAAQIscore.cpu().numpy(), haaqi_avg.squeeze(1).cpu().numpy()
            
            for i,j,k,l,m in zip(name,haaqi_score,predict_haaqi_score,hltype,hl.cpu().numpy()):
                print(i,j,k,l,m, sep=',',file=output_file) 
        output_file.close()
        
        test_total_loss = test_loss/len(test_loader) 
        test_total_haaqi = test_haaqi/len(test_loader)
        
        # calculate Pearson_cc and draw figure
        df = pd.read_csv(output_path)
        # SRCC is the Spearman Rank Correlation Coefficent
        # LCC is the normal Linear Correlation Coefficient
        haaqi = df['TrueHAAQI'].astype('float32').to_numpy()
        predict_haaqi = df['PredictHAAQI'].astype('float32').to_numpy()
        srcc1, pvalue = scipy.stats.spearmanr(haaqi, predict_haaqi)
        Pearson_cc1, p_value = scipy.stats.pearsonr(haaqi, predict_haaqi)
        mse1 = mean_squared_error(haaqi, predict_haaqi)
        
        plt.clf() # clear the fig before we draw
        fig, axs = plt.subplots(1, 2, figsize=(8,4))
        axs[0].plot(haaqi,predict_haaqi, 'o', ms=5)
        axs[0].tick_params(labelsize=8)
        axs[0].set_xlabel("True HAAQI")
        axs[0].set_ylabel("Predicted HAAQI")
        axs[0].set_title(f'LCC:{Pearson_cc1:5f},SRCC:{srcc1:5f}')
          
        Pearson_cc = (Pearson_cc1)
        plt.tight_layout()
        plt.savefig(f'{args.result_dir}scatter_{mode}.png')
        plt.close()
        
        print(f'HAAQI Test Loss:{test_total_haaqi:.5}, Pearson_cc:{Pearson_cc1:.6}, SRCC:{srcc1:.6}, MSE:{mse1}')
        
    torch.cuda.empty_cache()
    gc.collect()  

    end_time = time.time()
    total_time = end_time - start_time
    average_time_per_data = total_time / len(test_loader.dataset)
    print(f"Total time: {total_time:.2f} seconds, Average time per data: {average_time_per_data:.6f} seconds per data")
     
    return test_total_loss, Pearson_cc
