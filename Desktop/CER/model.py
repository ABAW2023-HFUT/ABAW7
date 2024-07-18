import torch
import os

import torch.nn as nn

from pre_model.posterV2 import ret_posterV2
from pre_model.resnet50 import ret_resnet50


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_emb = 0.5):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop_emb = nn.Dropout(drop_emb)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.drop_emb(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        
        return x

class CEF(torch.nn.Module):
    def __init__(self, lengths):
        super(CEF, self).__init__()
        self.posterV2 = ret_posterV2() # 768
        self.resnet50_1, self.resnet50_2 = ret_resnet50() # 2048
        
        self.cls = MLP(lengths, 1024, 7)
        self.head = MLP(lengths, 1024, lengths)
        self.FERhead = MLP(lengths, 1024, 6)
        

    def forward(self, x):
        B = x.shape[0]
        feat1 = self.posterV2(x)
        feat2 = (self.resnet50_1(x) + self.resnet50_2(x))/2
        
        # feat = feat2
        feat = torch.cat([feat1,feat2], dim=1)
        
        a,b,c = feat[:B//3,:],feat[B//3:B//3*2,:],feat[B//3*2:,:]
        
        return nn.Sigmoid()(self.cls(a)),nn.Sigmoid()(self.FERhead(a)),self.head(b),self.head(c)

class anoCEF(torch.nn.Module):
    def __init__(self, lengths):
        super(anoCEF, self).__init__()
        self.posterV2 = ret_posterV2() # 768
        self.resnet50_1, self.resnet50_2 = ret_resnet50() # 2048
        
        self.cls = MLP(lengths, 2048, 7)
        self.head = MLP(lengths, 2048, lengths)
        self.FERhead = MLP(lengths, 2048, 6)          
        

    def forward(self, x):
        B = x.shape[0]
        feat1 = self.posterV2(x)
        feat2 = (self.resnet50_1(x) + self.resnet50_2(x))/2
        
        # feat = feat2
        feat = torch.cat([feat1,feat2], dim=1)
        
        a,b,c = feat[:B//3,:],feat[B//3:B//3*2,:],feat[B//3*2:,:]
        
        return nn.Sigmoid()(self.cls(a)),nn.Sigmoid()(self.FERhead(a)),self.head(b),self.head(c)
    
    
class oldCEF(torch.nn.Module):
    def __init__(self, lengths):
        super(oldCEF, self).__init__()
        self.posterV2 = ret_posterV2() # 768
        self.resnet50_1, self.resnet50_2 = ret_resnet50() # 2048
        
        self.cls = nn.Linear(lengths, 7)
        self.head = MLP(lengths, 2048, lengths)
        # self.FERhead = MLP(lengths, 1024, 6)
        

    def forward(self, x):
        B = x.shape[0]
        feat1 = self.posterV2(x)
        feat2 = (self.resnet50_1(x) + self.resnet50_2(x))/2
        
        # feat = feat2
        feat = torch.cat([feat1,feat2], dim=1)
        
        a,b,c = feat[:B//3,:],feat[B//3:B//3*2,:],feat[B//3*2:,:]
        
        return self.cls(a),self.head(b),self.head(c)
    
    
if __name__ == '__main__':
    model = CEF(2048+768)