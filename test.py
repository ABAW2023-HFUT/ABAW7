import torch
from dataloader import Dataset
import torch.utils.data as data
import pandas as pd
import os, torch
import cv2
from torchvision import transforms
import numpy as np
from tqdm import tqdm

import time
import timm
import timm.scheduler
from model import Model,MyModel,seperateModel

from sklearn.metrics import f1_score
from eval import *
from utils import setup_seed, Logger, get_loss_fn, get_eval_fn, flatten_pred_label,CCC_loss,compute_VA_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
setup_seed(32)#42   24  32
device = 'cuda:0'


if __name__ == '__main__':
    # files = os.listdir('/data/shenkang/data/ABAW/features')
    
    
    feat_folder = '/data/shenkang/data/ABAW/features'
    file_length = {
    'posterv2_caer.npy':        768,
    #  'Ada-CM_res18.npy':        512,
     'LNS_swin_t.npy':          768,
    #  'vggface2.npy':            2048,
    #  'eacres.npy':              2048,
     'posterv2_raf.npy':        768,
     'posterv2_affectnet7.npy': 768,
    #  'ASM.npy':                 2048,
     'posterv2_affectnet8.npy': 768,
     "23_resnet18.npy":   512,
     "23_poster_rafdb.npy": 768,
     "ME-GraphAU_swin_tiny.npy":384,
     "23_fau.npy":17,
     "ASM_resnet18_112.npy":512
     }
    
    # files = list(file_length.keys())[11:12]
    # lengths = list(file_length.values())[11:12]
    
    files = ['23_poster2_affect8.npy','ASM_resnet18.npy',"23_fau.npy","posterv2_affectnet8.npy","23_resnet18.npy"]
    lengths = [768,512,17,768,512]
    
    # feat_folder = '/data/shenkang/ABAW/feature2023/Aff-Wild2/features'
    # files = ['poster2_rafdb.npy','fau.npy']
    # lengths = [768,17]
    
    
    print('-' * 50)
    print('load features : {}'.format(', '.join(files)))
    print('-' * 50)
    
    a = time.time()
    
    train_dataset = Dataset(path = '/data/shenkang/data/ABAW/Annotations', phase = 'train',  feats_paths = [os.path.join(feat_folder,file) for file in files], feat_aug=True)
    train_dataloader = data.DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8)
    
    print(time.time() - a)
    
    val_dataset = Dataset(path = '/data/shenkang/data/ABAW/Annotations', phase = 'val',  feats_paths = [os.path.join(feat_folder,file) for file in files])
    val_dataloader = data.DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=8)
    
    
    model1 = MyModel(lengths,
                    input_dim=384,
                    mlp_ratio=4,
                    nheads=4,
                    nlayer=4,
                    dropout=0.5,
                    cls=True,
                    out_head='linear').cuda()
    model2 = MyModel(lengths,
                    input_dim=384,
                    mlp_ratio=4,
                    nheads=4,
                    nlayer=4,
                    dropout=0.5,
                    cls=True,
                    out_head='linear').cuda()
    model3 = MyModel(lengths,
                    input_dim=384,
                    mlp_ratio=4,
                    nheads=4,
                    nlayer=4,
                    dropout=0.5,
                    cls=True,
                    out_head='linear').cuda()
    model4 = MyModel(lengths,
                    input_dim=384,
                    mlp_ratio=4,
                    nheads=4,
                    nlayer=4,
                    dropout=0.5,
                    cls=True,
                    out_head='linear').cuda()
    
    # model = seperateModel(lengths,
    #                 input_dim=384,
    #                 mlp_ratio=4,
    #                 nheads=4,
    #                 nlayer=2,
    #                 dropout=0.5,
    #                 cls=True,
    #                 out_head='linear').cuda()

    # model = torch.nn.DataParallel(model,device_ids=[0,1])
    
    
    optimizer = torch.optim.AdamW([{'params': model1.parameters()},
                                  {'params': model2.parameters()},
                                  {'params': model3.parameters()},
                                  {'params': model4.parameters()}],lr = 1e-4)
    # scheduler = timm.scheduler.CosineLRScheduler(optimizer,
    #                                              t_initial=100,
    #                                              lr_min=1e-5,
    #                                              warmup_t=10,
    #                                              warmup_lr_init=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # optimizer = torch.optim.AdamW(model.parameters(),lr = 2.5e-5)
    # scheduler = timm.scheduler.CosineLRScheduler(optimizer,
    #                                              t_initial=100,
    #                                              lr_min=2.5e-6,
    #                                              warmup_t=10,
    #                                              warmup_lr_init=2.5e-6)
    
    
    loss_V = get_loss_fn("V", device)
    loss_A = get_loss_fn("A", device)
    loss_EXPR = get_loss_fn("EXPR", device)
    loss_AU = get_loss_fn("AU", device)
    V_max=0.48
    A_max=0.42
    F1_EXPR_max=0.30
    AU_max=0.49
    for epoch in tqdm(range(50)):
        scheduler.step(epoch)
        print(f'epoch {epoch} learning rate:' + str(optimizer.param_groups[0]['lr']))
        
        total_loss = [0,0,0,0]
        num = [0,0,0,0]
        
        model1.train()
        model2.train()
        model3.train()
        model4.train()
        for batch,(features, valence, arousal, expression, aus) in enumerate(train_dataloader):
            valence, arousal, expression, aus = valence.cuda(), arousal.cuda(), expression.cuda(), aus.cuda()
            
            for i in range(len(features)):
                features[i] = features[i].cuda()
            
            
            B = valence.shape[0]
            
            preds1 = model1(features)
            preds2 = model2(features)
            preds3 = model3(features)
            preds4 = model4(features)
            pred_V = preds1[0].squeeze()
            pred_A = preds2[1].squeeze()
            pred_FER = preds3[2]
            pred_AU = torch.cat(preds4[3:],dim=1)
            
            mask1 = (valence > -4).float()
            mask2 = (arousal > -4).float()
            mask3 = (expression != -1).float()
            mask4 = (aus != -1).float()
            # mask1 = torch
            
            
            
            loss1 = 0.6*(loss_V(pred_V,valence) * mask1).sum() / mask1.sum()+loss_A(pred_V,valence)*0.4 
            
            # loss1,_,_=compute_VA_loss(pred_V,pred_A,valence,arousal)
            # loss2=CCC_loss(pred_A,arousal)
            loss2 = 0.5*loss_A(pred_A,arousal)+ (loss_V(pred_A,arousal) * mask2).sum() / mask2.sum()*0.5
            loss3 = (loss_EXPR(pred_FER,expression) * mask3).sum() / mask3.sum()
            loss4 = (loss_AU(pred_AU , aus ))
                    
            loss = (loss1 + loss2)/2+ loss3 + loss4*2
            # loss=loss3
            # loss = loss3 / 3
            # loss = (loss1 + loss2)/2 + loss3
            
            # print('V_loss:{:.4f}\tA_loss:{:.4f}\tFER_loss:{:.4f}\tAU_loss:{:.4f}'.format(loss1.item(),loss2.item(),loss3.item(),loss4.item()))
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss[0] += loss1.item() * mask1.sum()
            total_loss[1] += loss2.item() * mask2.sum()
            total_loss[2] += loss3.item() * mask3.sum()
            total_loss[3] += loss4.item() * mask4.sum()
            
            num[0] += mask1.sum()
            num[1] += mask2.sum()
            num[2] += mask3.sum()
            num[3] += mask4.sum()
            
            
        
        # if(epoch < 10):
        #     continue
        print('-'*50)
        print('start validation...')
        val_total_loss = [0,0,0,0]
        val_num = [0,0,0,0]
        
        total_preds = [[],[],[],[]]
        total_gt = [[],[],[],[]]
        with torch.no_grad():
            model1.eval()
            model2.eval()
            model3.eval()
            model4.eval()
            for batch,(features, valence, arousal, expression, aus) in enumerate(val_dataloader):
                valence, arousal, expression, aus = valence.cuda(), arousal.cuda(), expression.cuda(), aus.cuda()
                for i in range(len(features)):
                    features[i] = features[i].cuda()
                    
                preds1 = model1(features)
                preds2 = model2(features)
                preds3 = model3(features)
                preds4 = model4(features)
                
                pred_V = preds1[0].squeeze()
                pred_A = preds2[1].squeeze()
                pred_FER = preds3[2]
                pred_AU = torch.cat(preds4[3:],dim=1)
                # print("pred_V",pred_V)
                # print("pred_A",pred_A)
                # print("pred_FER",pred_FER)
                # print("pred_AU",pred_AU)
                # if(epoch > 5):
                #     a = 0
                #     b = 0
                #     for i in range(100):
                #         print("{:.4f},{:.4f},".format(pred_V[i].item() - valence[i].item(),pred_A[i].item() - arousal[i].item()))
                #         a += (pred_V[i].item() - valence[i].item())**2
                #         b += (pred_A[i].item() - arousal[i].item())**2
                        
                #     print(a,b)
                #     exit()
                    
                    
                
                
                mask1 = (valence > -4).float()
                mask2 = (arousal > -4).float()
                mask3 = (expression != -1).float()
                mask4 = (aus != -1).float()
                
                loss1 = (loss_V(pred_V,valence) * mask1).sum() / mask1.sum()
                # loss1,_,_=compute_VA_loss(pred_V,pred_A,valence,arousal)
                loss2 = (loss_A(pred_A,arousal) * mask2).sum() / mask2.sum()
                loss3 = (loss_EXPR(pred_FER,expression) * mask3).sum() / mask3.sum()
                loss4 = (loss_AU(pred_AU , aus ))
                
                val_total_loss[0] += loss1.item() * mask1.sum()
                val_total_loss[1] += loss2.item() * mask2.sum()
                val_total_loss[2] += loss3.item() * mask3.sum()
                val_total_loss[3] += loss4.item() * mask4.sum()
                
                val_num[0] += mask1.sum()
                val_num[1] += mask2.sum()
                val_num[2] += mask3.sum()
                val_num[3] += mask4.sum()
            
                total_preds[0].append(pred_V.cpu().detach())
                total_preds[1].append(pred_A.cpu().detach())
                total_preds[2].append(pred_FER.cpu().detach())
                total_preds[3].append(pred_AU.cpu().detach())
                
                total_gt[0].append(valence.cpu().detach())
                total_gt[1].append(arousal.cpu().detach())
                total_gt[2].append(expression.cpu().detach())
                total_gt[3].append(aus.cpu().detach())
        
            
        for i in range(4):
            total_preds[i] = torch.cat(total_preds[i],dim=0).numpy().copy()
            total_gt[i] = torch.cat(total_gt[i],dim=0).numpy().copy()
        
        
        
        
        
        # 掩码 测试集中VA和AU均不需要，训练集每一个都需要
        mask3 = (total_gt[2] != -1)
        total_preds[2],total_gt[2] = total_preds[2][mask3],total_gt[2][mask3]
        
        total_preds[2] = np.argmax(total_preds[2],axis=1)
        total_preds[3] = np.around(total_preds[3])
        
        V_CCC = VA_metric(total_preds[0],total_gt[0])
        A_CCC = VA_metric(total_preds[1],total_gt[1])
        
        F1_EXPR = f1_score(total_gt[2],total_preds[2],average= 'macro')
        F1_AU = f1_score(total_gt[3],total_preds[3],average= 'macro')
                
        print('Train set: V_loss:{:.4f}\tA_loss:{:.4f}\tFER_loss:{:.4f}\tAU_loss:{:.4f}'.format(total_loss[0] / num[0],total_loss[1] / num[1],total_loss[2] / num[2],total_loss[3] / num[3]))
        print('Validation set: V_loss:{:.4f}\tA_loss:{:.4f}\tFER_loss:{:.4f}\tAU_loss:{:.4f}'.format(val_total_loss[0] / val_num[0],val_total_loss[1] / val_num[1],val_total_loss[2] / val_num[2],val_total_loss[3] / val_num[3]))
        print('Validation set: V_CCC:{:.4f}\tA_CCC:{:.4f}\tEXPR_F1:{:.4f}\tAU_F1:{:.4f}'.format(V_CCC,A_CCC,F1_EXPR,F1_AU))
        print('-'*50)
        
        if V_max<V_CCC:
            V_max=V_CCC
            torch.save({'model1_state_dict': model1.state_dict()},
            'ckpt/V/{}.pth'.format('+'.join([file.split('.')[0] for file in files])+'+'+str(V_max)+'+'+str(epoch)))
        if A_max<A_CCC:
            A_max=A_CCC
            torch.save({'model2_state_dict': model2.state_dict()},
            'ckpt/A/{}.pth'.format('+'.join([file.split('.')[0] for file in files])+'+'+str(A_max)+'+'+str(epoch)))
        if F1_EXPR_max<F1_EXPR:
            F1_EXPR_max=F1_EXPR
            torch.save({'model3_state_dict': model3.state_dict()},
            'ckpt/EXPR/{}.pth'.format('+'.join([file.split('.')[0] for file in files])+'+'+str(F1_EXPR_max)+'+'+str(epoch)))
        if AU_max<F1_AU:
            AU_max=F1_AU
            torch.save({'model4_state_dict': model4.state_dict()},
            'ckpt/AU/{}.pth'.format('+'.join([file.split('.')[0] for file in files])+'+'+str(AU_max)+'+'+str(epoch)))
        
        loss_save_path = 'loss/'+'+'.join([file.split('.')[0] for file in files])
        if not os.path.exists(f'{loss_save_path}.csv'):
            with open(f'{loss_save_path}.csv','w') as f:
                f.write('epoch,train_V_loss,train_A_loss,train_FER_loss,train_AU_loss,val_V_loss,val_A_loss,val_FER_loss,val_AU_loss,V_CCC,A_CCC,F1_EXPR,F1_AU\n')
                f.write(f'{epoch},{total_loss[0] / num[0]},{total_loss[1] / num[1]},{total_loss[2] / num[2]},{total_loss[3] / num[3]},{val_total_loss[0] / val_num[0]},{val_total_loss[1] / val_num[1]},{val_total_loss[2] / val_num[2]},{val_total_loss[3] / val_num[3]},{V_CCC},{A_CCC},{F1_EXPR},{F1_AU}\n')
        else:
            with open(f'{loss_save_path}.csv','a') as f:
                f.write(f'{epoch},{total_loss[0] / num[0]},{total_loss[1] / num[1]},{total_loss[2] / num[2]},{total_loss[3] / num[3]},{val_total_loss[0] / val_num[0]},{val_total_loss[1] / val_num[1]},{val_total_loss[2] / val_num[2]},{val_total_loss[3] / val_num[3]},{V_CCC},{A_CCC},{F1_EXPR},{F1_AU}\n')
            
            
    torch.save({'model1_state_dict': model1.state_dict(),
                            'model2_state_dict': model2.state_dict(),
                            'model3_state_dict': model3.state_dict(),
                            'model4_state_dict': model4.state_dict()},
            'ckpt/{}.pth'.format('+'.join([file.split('.')[0] for file in files])+'+'+str(epoch)))
            
            
            
            
# def test(dataloader,model):
#     print('-'*50)
#     print('start validation...')
#     model.eval()
#     val_total_loss = [0,0,0,0]
#     val_num = [0,0,0,0]
#     total_preds = [[],[],[],[]]
#     total_gt = [[],[],[],[]]
#     with torch.no_grad():
#         model.eval()
#         for batch,(features, valence, arousal, expression, aus) in enumerate(dataloader):
#             valence, arousal, expression, aus = valence.cuda(), arousal.cuda(), expression.cuda(), aus.cuda()
#             for i in range(len(features)):
#                 features[i] = features[i].cuda().float()
            
#             preds = []
#             for i in range(len(model)):
#                 pred = model[i](features)
#                 preds.append(pred)
            
#             pred_V = preds[0].squeeze()
#             pred_A = preds[1].squeeze()
#             pred_FER = preds[2]
#             pred_AU = torch.cat(preds[3:],dim=1)
                
#             mask1 = (valence > -4).float()
#             mask2 = (arousal > -4).float()
#             mask3 = (expression != -1).float()
#             mask4 = (aus != -1).float()
            
            
#             # _,loss1,_ = (compute_VA_loss(pred_V,pred_A,valence,arousal))
#             _,_,loss2 = (compute_VA_loss(pred_V,pred_A,valence,arousal))
#             loss1 = ((loss_VA(pred_V,valence) * mask1).sum() / mask1.sum() + (loss_VA(pred_A,arousal) * mask2).sum() / mask2.sum())/2
#             # loss2 = (loss_VA(pred_A,arousal) * mask2).sum() / mask2.sum()
#             loss3 = (loss_EXPR(pred_FER,expression) * mask3).sum() / mask3.sum()
#             # loss4 = (loss_AU(pred_AU * mask4, aus * mask4)).sum() / mask4.sum()
#             loss4 = loss_AU(pred_AU, aus)
            
#             val_total_loss[0] += loss1.item() * mask1.sum()
#             val_total_loss[1] += loss2.item() * mask2.sum()
#             val_total_loss[2] += loss3.item() * mask3.sum()
#             val_total_loss[3] += loss4.item() * mask4.sum()
            
#             val_num[0] += mask1.sum()
#             val_num[1] += mask2.sum()
#             val_num[2] += mask3.sum()
#             val_num[3] += mask4.sum()
        
#             total_preds[0].append(pred_V.cpu().detach())
#             total_preds[1].append(pred_A.cpu().detach())
#             total_preds[2].append(pred_FER.cpu().detach())
#             total_preds[3].append(pred_AU.cpu().detach())
            
#             total_gt[0].append(valence.cpu().detach())
#             total_gt[1].append(arousal.cpu().detach())
#             total_gt[2].append(expression.cpu().detach())
#             total_gt[3].append(aus.cpu().detach())
    
        
            
        
#     for i in range(4):
#         total_preds[i] = torch.cat(total_preds[i],dim=0).numpy().copy()
#         total_gt[i] = torch.cat(total_gt[i],dim=0).numpy().copy()
    
#     # 掩码 测试集中VA和AU均不需要，训练集每一个都需要
#     mask3 = (total_gt[2] != -1)
#     total_preds[2],total_gt[2] = total_preds[2][mask3],total_gt[2][mask3]
    
#     total_preds[2] = np.argmax(total_preds[2],axis=1)
#     total_preds[3] = np.around(total_preds[3])
    
    
#     V_CCC = VA_metric(total_preds[0],total_gt[0])
#     A_CCC = VA_metric(total_preds[1],total_gt[1])

    
#     F1_EXPR = f1_score(total_gt[2],total_preds[2],average= 'macro')
#     F1_AU = f1_score(total_gt[3],total_preds[3],average= 'macro')
    
#     # print('Train set: V_loss:{:.4f}\tA_loss:{:.4f}\tFER_loss:{:.4f}\tAU_loss:{:.4f}'.format(total_loss[0] / num[0],total_loss[1] / num[1],total_loss[2] / num[2],total_loss[3] / num[3]))
#     print('Validation set: V_loss:{:.4f}\tA_loss:{:.4f}\tFER_loss:{:.4f}\tAU_loss:{:.4f}'.format(val_total_loss[0] / val_num[0],val_total_loss[1] / val_num[1],val_total_loss[2] / val_num[2],val_total_loss[3] / val_num[3]))
#     print('Validation set: V_CCC:{:.4f}\tA_CCC:{:.4f}\tEXPR_F1:{:.4f}\tAU_F1:{:.4f}'.format(V_CCC,A_CCC,F1_EXPR,F1_AU))
#     print('-'*50) 
    
    