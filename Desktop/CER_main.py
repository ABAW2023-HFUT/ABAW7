import torch
import torch.utils.data as data
import pandas as pd
import os, torch

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import cv2
from torchvision import transforms
import numpy as np
from tqdm import tqdm

import time
import timm
import timm.scheduler

import random
from CEF.loss import *
from CEF.model import *
from CEF.dataloader import Dataset
from sklearn.metrics import f1_score
from pre_model.transformer import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"


# device = torch.device('cuda:7')

def test(model, test_dataloader):
    preds = []
    gts = []
    total_loss = 0
    total_num = 0
    print("-"*80)
    print("evaluate...")
    print("-"*80)
    with torch.no_grad():
        for batch, (img_ori, img_1, img_2, CE, FER) in enumerate(test_dataloader):
            CE,FER = CE.cuda(),FER.cuda()
            imgs = torch.cat([img_ori,img_1,img_2],dim=0).cuda()

            pred_ce,pred_fer,vector1,vector2 = model(imgs)

            fer2ce_0 = (pred_fer[:,2] + pred_fer[:,5]) / 2
            fer2ce_1 = (pred_fer[:,3] + pred_fer[:,5]) / 2
            fer2ce_2 = (pred_fer[:,4] + pred_fer[:,5]) / 2
            fer2ce_3 = (pred_fer[:,1] + pred_fer[:,5]) / 2
            fer2ce_4 = (pred_fer[:,0] + pred_fer[:,5]) / 2
            fer2ce_5 = (pred_fer[:,2] + pred_fer[:,4]) / 2
            fer2ce_6 = (pred_fer[:,0] + pred_fer[:,4]) / 2
            
            
            
            pred_ce = (pred_ce + torch.stack([fer2ce_0, fer2ce_1,fer2ce_2,fer2ce_3,fer2ce_4,fer2ce_5,fer2ce_6],dim=1))/2
            
            mask = (CE != -1).float()
            total_loss += (CEloss(pred_ce,CE) * mask).sum()
            total_num += mask.sum()
            
            pred_ce = pred_ce.detach().cpu().numpy()
            
            total_preds = np.argmax(pred_ce,axis=1)
            for i in range(len(total_preds)):
                if(CE[i] != -1):
                    preds.append(total_preds[i])
                    gts.append(CE[i].item())
        print("loss : ",(total_loss/total_num).item(  ))
        print("f1_score : ",f1_score(np.array(gts), np.array(preds), average= 'macro'))


import time

if __name__ =='__main__':
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # time.sleep(2000)
    
    
    train_dataset = Dataset(phase = 'train',add_dataset=['c','raf'])
    train_dataloader = data.DataLoader(train_dataset, batch_size=96, shuffle=True, num_workers=4)
    
    test_dataset = Dataset(phase = 'test',add_dataset=['c','raf'])
    test_dataloader = data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    model = anoCEF(2048+768).cuda()
    
    model = nn.DataParallel(model, device_ids=[0,1]).cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer = torch.optim.Adam([{'params':model.module.posterV2.parameters(),'lr' : 3e-6},
                                  {'params':model.module.resnet50_1.parameters(),'lr' : 3e-6},
                                  {'params':model.module.resnet50_2.parameters(),'lr' : 3e-6},
                                  {'params':model.module.cls.parameters(),'lr' : 1e-5},
                                  {'params':model.module.head.parameters(),'lr' : 1e-5},
                                  {'params':model.module.FERhead.parameters(),'lr' : 1e-5}])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    max_batch = len(train_dataset)//96
    
    
    step = 0
    for epoch in tqdm(range(50)):
        
        for batch, (img_ori, img_1, img_2, CE, FER) in enumerate(train_dataloader):
            if batch == max_batch:
                continue
            try:
                CE,FER = CE.cuda(),FER.cuda()
                imgs = torch.cat([img_ori,img_1,img_2],dim=0).cuda()

                pred_ce,pred_fer,vector1,vector2 = model(imgs)
                
                
                    
                fer2ce_0 = (pred_fer[:,2] + pred_fer[:,5]) / 2
                fer2ce_1 = (pred_fer[:,3] + pred_fer[:,5]) / 2
                fer2ce_2 = (pred_fer[:,4] + pred_fer[:,5]) / 2
                fer2ce_3 = (pred_fer[:,1] + pred_fer[:,5]) / 2
                fer2ce_4 = (pred_fer[:,0] + pred_fer[:,5]) / 2
                fer2ce_5 = (pred_fer[:,2] + pred_fer[:,4]) / 2
                fer2ce_6 = (pred_fer[:,0] + pred_fer[:,4]) / 2
                
                
                
                pred_ce = (pred_ce + torch.stack([fer2ce_0, fer2ce_1,fer2ce_2,fer2ce_3,fer2ce_4,fer2ce_5,fer2ce_6],dim=1))/2
                
                if(pred_ce.shape[0] != CE.shape[0]):
                    print(pred_ce.shape[0])
                    print(pred_fer.shape[0])
                    print(CE.shape[0])
                    optimizer.zero_grad()
                    continue
                
                mask_ce = (CE != -1).float()
                mask_fer = (FER != -1).float()
                
                if(mask_ce.sum() == 0):
                    celoss = torch.tensor([0]).cuda()
                else:
                    celoss = (CEloss(pred_ce,CE) * mask_ce).sum() / mask_ce.sum()
                
                if(mask_fer.sum() == 0):
                    ferloss = torch.tensor([0]).cuda()
                else:
                    ferloss = (CEloss(pred_fer,FER) * mask_fer).sum() / mask_fer.sum()
                
                cosloss = ContrastiveLoss(vector1,vector2,device=next(model.parameters()).device)
                
                loss = 3 * celoss + ferloss + cosloss
                
                if(batch%50 == 0):
                    print(str(batch)+'/'+str(max_batch),':','celoss:{:.4f}'.format(celoss.item()),'ferloss:{:.4f}'.format(ferloss.item()),'cosloss:{:.4f}'.format(cosloss.item()))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        
           
                    
                step += 1
            except Exception as e:
                print(e)
                print("-"*50)
                print(pred_ce.shape[0])
                print(pred_fer.shape[0])
                print(CE.shape[0])
                print()
                optimizer.zero_grad()
                continue
            
        test(model, test_dataloader)
        scheduler.step()
        torch.save(model.state_dict(), './checkpoint/2CEF{}.pth'.format(epoch))
    torch.save(model.state_dict(), './ckpt/CEF_final.pth')        
            
        
        
        
            
            
            
                    
        