import torch
import torch.nn as nn
import torch.nn.functional as F

def CEloss(pred,gt,smooth=0):
    criterion = nn.CrossEntropyLoss(reduction='none',ignore_index=-1,label_smoothing=smooth)
    loss = criterion(pred,gt)
    return loss

    
def ContrastiveLoss(vectors1,vectors2, device, margin=1.0):
    # 效果同上
    vectors1 = nn.functional.normalize(vectors1, dim=1)
    vectors2 = nn.functional.normalize(vectors2, dim=1)
    
    batchsize = vectors1.shape[0]
    
    euclidean_distance = torch.cdist(vectors1, vectors2, p=2) # 计算所有的距离

    positive_pairs_distance = euclidean_distance.diag() # 对角线
    
    mask = torch.ones_like(euclidean_distance) - torch.eye(batchsize).to(device) # 对角线全0，其他全1
    negative_pairs_distance = euclidean_distance + (1 - mask) * margin
    
    positive_loss = torch.sum(torch.pow(positive_pairs_distance, 2)) / batchsize
    negative_loss = torch.sum(torch.clamp(margin - negative_pairs_distance, min=0.0) ** 2) / (batchsize * (batchsize - 1))
        
    loss = positive_loss + negative_loss
    return loss