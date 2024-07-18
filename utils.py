import torch
import numpy as np
import random
import sys
from eval import VA_metric, EXPR_metric, AU_metric
import torch.nn as nn
import torch.nn.functional as F
class WeightedAsymmetricLoss(nn.Module):
    def __init__(self, device, eps=1e-8, disable_torch_grad=True, weight=None):
        super(WeightedAsymmetricLoss, self).__init__()
        self.disable_torch_grad = disable_torch_grad
        self.eps = eps
        self.weight = weight
        self.device = device

    def forward(self, x, y):
        
        mask = (y != -1).float()
        # print(mask)
        xs_pos = x
        xs_neg = 1 - x

        # Basic CE calculation
        # clamp 最小值变为min，最大值变为max
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.disable_torch_grad:
            torch.set_grad_enabled(False)
        neg_weight = 1 - xs_neg
        if self.disable_torch_grad:
            torch.set_grad_enabled(True)
        loss = los_pos + neg_weight * los_neg
        
        
        loss = loss * mask
        
        loss = loss.to(self.device)
        # pdb.set_trace()
        # print('loss',str(loss.data))
        # print('weight',str(self.weight.data))
        if self.weight is not None:

            loss = loss * self.weight.view(1,-1)

        

        # loss = loss.to(self.device)
        loss = loss.mean(dim=-1)

        # print(loss.shape)
        return -torch.sum(loss) / torch.sum(mask)
    

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Logger(object):
    def __init__(self, log_file="log_file.log"):
        self.terminal = sys.stdout
        self.file = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.file.flush()

def get_annotations(task):
    annotations = {
        'VA':'VA_Estimation_Challenge',
        'EXPR':'EXPR_Classification_Challenge',
        'AU':'AU_Detection_Challenge'
    }
    return annotations[task]

def get_disregard(task):
    disregard = {
        'VA':-5,
        'EXPR':-1,
        'AU':-1
    }
    return disregard[task]

def get_activation(task):
    if task == 'VA':
        return nn.Tanh()
    elif task == 'EXPR':
        return nn.Sigmoid()
    elif task == 'AU':
        return nn.Sigmoid()

def get_num_output(task):
    num_output = {
        'VA':1,
        'EXPR':8,
        'AU':1
    }
    return num_output[task]

def get_num_linear(task):
    num_linear = {
        'VA':2,
        'EXPR':1,
        'AU':12
    }
    return num_linear[task]

def get_loss_fn(task, device):
    if task == 'V':
        return nn.MSELoss(reduction='none')
        # return CCCLoss(1)
    elif task=='A':
        return CCCLoss(1)
    elif task == 'EXPR':
        return nn.CrossEntropyLoss(reduction='none',ignore_index=-1)
    elif task == 'AU':
        train_weight = torch.from_numpy(np.loadtxt('weight.txt'))
        # train_weight = torch.tensor([0.7,0.2,0.1])
        train_weight = train_weight.to(device)
        return WeightedAsymmetricLoss(device, weight=train_weight)
        # return nn.BCEWithLogitsLoss(reduction='none')

def get_eval_fn(task):
    if task == 'VA':
        return VA_metric
    elif task == 'EXPR':
        return EXPR_metric
    elif task == 'AU':
        return AU_metric

def flatten_pred_label(lst):
    return np.concatenate([np.array(l) for l in lst])

def get_num_label(task):
    num_label = {
        'VA':2,
        'EXPR':1,
        'AU':12
    }
    return num_label[task]


def CCC_loss(x, y):
    indices_to_remove = (y != -5).nonzero(as_tuple=True)
    x, y = x.index_select(0, indices_to_remove[0]), y.index_select(0, indices_to_remove[0])
    
    vx = x - torch.mean(x) 
    vy = y - torch.mean(y) 
    rho =  torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2)))+1e-8)
    x_m = torch.mean(x)
    y_m = torch.mean(y)
    x_s = torch.std(x)
    y_s = torch.std(y)
    ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
    return 1-ccc

def compute_VA_loss(Vout,Aout,label_V,label_A):
    ccc_loss = CCC_loss(Vout,label_V) + CCC_loss(Aout,label_A)
    mse_loss = nn.MSELoss()(Vout,label_V) + nn.MSELoss()(Aout,label_A)
    
    loss = ccc_loss/2
    return loss,mse_loss,ccc_loss

from torch.autograd import Variable

class CCCLoss(nn.Module):
    def __init__(self, digitize_num, range=[-1, 1], eps=1e-8):
        super(CCCLoss, self).__init__() 
        self.digitize_num =  digitize_num
        self.range = range
        self.eps=eps
        if self.digitize_num !=0:
            bins = np.linspace(*self.range, num= self.digitize_num)
            self.bins = Variable(torch.as_tensor(bins, dtype = torch.float32).cuda()).view((1, -1))

    def forward(self, x, y): 
        indices_to_remove = (y != -5).nonzero(as_tuple=True)
        x, y = x.index_select(0, indices_to_remove[0]), y.index_select(0, indices_to_remove[0])
    
        y = y.view(-1)
        if self.digitize_num !=1:
            x = F.softmax(x, dim=-1)
            x = (self.bins * x).sum(-1)
        x = x.view(-1)
        vx = x - torch.mean(x) 
        vy = y - torch.mean(y) 
        rho =  torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2)))+self.eps)
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
        return 1-ccc
    


if __name__ == '__main__':
    # gt = torch.tensor([[1,1,0],[0,1,1],[-1,1,-1],[-1,-1,-1]])
    # pred = torch.tensor([[0,1,1],[0,1,1],[0,1,1],[0,1,0]])
    # print(get_loss_fn('AU',gt.device)(pred,gt))
    gt = torch.tensor([[0.89,1],[0.1,0.1],[.0,.0],[-5,-5]])
    pred = torch.tensor([[0.3,0.8],[0.1,0.2],[0.4,0.5],[0,0.2]])
    
    label_A = gt[:,1]
    label_V = gt[:,0]
    
    pred_A = pred[:,1]
    pred_V = pred[:,0]
    print(compute_VA_loss(pred_V,pred_A,label_V,label_A))
    
    