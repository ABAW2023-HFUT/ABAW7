import argparse
import os
import sys
from datetime import datetime
from dateutil import tz
from utils import setup_seed, Logger, get_loss_fn, get_eval_fn, flatten_pred_label
from dataset import AffWild2_Dataset
from torch.utils.data import DataLoader
from model import Model
import torch
import torch.optim as optim
from time import time
from dataloader import Dataset
import torch.utils.data as data
def parse_args():
    parser = argparse.ArgumentParser(description='ABAW 2023')
    parser.add_argument('--task', type=str, default='EXPR', choices=['VA', 'EXPR', 'AU'],
                        help='Specify the task (VA, EXPR, AU).')
    parser.add_argument('--feature', nargs='+', default=['eac'], 
                        help="Specify the features used")
    parser.add_argument('--model', type=str, default='Transformer', choices=['Transformer', 'TEMMA', 'LTSF'],
                        help="Specify the model used")
    parser.add_argument('--root', type=str, default='/data/shenkang/ABAW/feature2023/Aff-Wild2',
                        help="Specify the dataset root directory")
    parser.add_argument('--epochs', type=int, default=999,
                        help='Specify the number of epochs.')
    parser.add_argument('--early_stopping_patience', type=int, default=15, 
                        help='Patience for early stopping')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Specify the batch size.')
    parser.add_argument('--seq_step', type=int, default=256,
                        help='Specify the sequence step length')
    parser.add_argument('--seq_len', type=int, default=256,
                        help='Specify the sequence length')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Specify initial learning rate.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Specify the initial random seed (default: 0).')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Specify the number of workers to use')
    # Transformer settings
    parser.add_argument('--t_dropout', type=float, default=0.2)
    parser.add_argument('--d_affine_dim', type=int, default=512)
    # parser.add_argument('--d_feedforward', type=int, default=1024)
    parser.add_argument('--t_nheads', type=int, default=4)
    parser.add_argument('--t_nlayer', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--dropout_embed', type=float, default=0.5)
    parser.add_argument('--gpu', type=int, default=0, help='the number of the device')
    args = parser.parse_args()
    return args

def main(args):
    args.d_affine_dim = 512 if args.task == 'VA' else 256
    setup_seed(args.seed)
    device = torch.device('cuda:{}'.format(args.gpu))
    print('Loading data......')
    # train_dataset = AffWild2_Dataset(args, phase='train')
    # val_dataset = AffWild2_Dataset(args, phase='val')
    # print('Train dataset size: ', len(train_dataset))
    # print('Val dataset size: ', len(val_dataset))

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    train_dataset = Dataset(path = '/data/shenkang/data/ABAW/Annotations', phase = 'train',  feats_paths = ['/data/shenkang/ABAW/feature2023/Aff-Wild2/features/eac.npy'])
    train_loader = data.DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8)
    
    
    val_dataset = Dataset(path = '/data/shenkang/data/ABAW/Annotations', phase = 'val',  feats_paths = ['/data/shenkang/ABAW/feature2023/Aff-Wild2/features/eac.npy'])
    val_loader = data.DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=8)


    args.feat_dim = [2048]
    # args.d_in = sum(train_dataset.get_feature_dim())
    model = Model(args)

    model = model.to(device)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_fn = get_loss_fn(args.task, device)
    eval_fn = get_eval_fn(args.task)
    if args.task != 'VA':
        best_val_score = 0
        best_val_epoch = 0
        early_stop = 0
        for epoch in range(1, args.epochs+1):
            start = time()
            train_loss, train_score = train(model, train_loader, optimizer, loss_fn, eval_fn, device)
            scheduler.step()
            val_loss, val_score = evaluate(model, val_loader, loss_fn, eval_fn, device)
            end = time()
            print("Epoch: [{:03d}/{:03d}], train loss: {:.4f}, train_score: {:.4f}".format(
                                    epoch, args.epochs, train_loss, train_score))
            print("Epoch: [{:03d}/{:03d}], val loss: {:.4f}, val_score: {:.4f}, time: {:.2f}s".format(
                                    epoch, args.epochs, val_loss, val_score, end-start))
            if val_score - best_val_score >= 0.0001:
                early_stop = 0
                best_val_score = val_score
                best_val_epoch = epoch
                print('Saving checkpoint: {}'.format(epoch))
                state_dict = model.state_dict()
                torch.save({'epoch': epoch,
                            'state_dict': state_dict},
                            args.model_path)
            else:
                early_stop += 1
                if early_stop >= args.early_stopping_patience:
                    print(f'Note: target can not be optimized for {args.early_stopping_patience} consecutive epochs, '
                        f'early stop the training process!')
                    break
            print('-' * 50)
        print('Finally Best Score: {:.4f} in epoch: {}'.format(best_val_score, best_val_epoch))
        print('finishing training')
    else:
        best_val_v = 0
        best_val_a = 0
        accp_val_v = 0
        accp_val_a = 0
        best_val_epoch = 0
        early_stop = 0
        for epoch in range(1, args.epochs+1):
            start = time()
            train_loss, (train_v, train_a) = train(model, train_loader, optimizer, loss_fn, eval_fn, device)
            scheduler.step()
            val_loss, (val_v, val_a) = evaluate(model, val_loader, loss_fn, eval_fn, device)
            end = time()
            print("Epoch: [{:03d}/{:03d}], train loss: {:.4f}, train_v: {:.4f}, train_a: {:.4f}, train_va: {:.4f}".format(
                                    epoch, args.epochs, train_loss, train_v, train_a, (train_a+train_v)/2))
            print("Epoch: [{:03d}/{:03d}], val loss: {:.4f}, val_v: {:.4f}, val_a: {:.4f}, val_va: {:.4f}, time: {:.2f}s".format(
                                    epoch, args.epochs, val_loss, val_v, val_a, (val_v+val_a)/2, end-start))
            if val_v - best_val_v >= 0.0001:
                early_stop = 0
                best_val_v = val_v
                accp_val_a = val_a
                best_val_epoch = epoch
                print('Saving v: {}'.format(epoch))
                state_dict = model.state_dict()
                torch.save({'epoch': epoch,
                            'state_dict': state_dict},
                            args.model_path[0])
            if val_a - best_val_a >= 0.0001:
                early_stop = 0
                best_val_a = val_a
                accp_val_v = val_v
                best_val_epoch = epoch
                print('Saving a: {}'.format(epoch))
                state_dict = model.state_dict()
                torch.save({'epoch': epoch,
                            'state_dict': state_dict},
                            args.model_path[1])
            if val_v - best_val_v < 0.0001 and val_a - best_val_a < 0.0001:
                early_stop += 1
                if early_stop >= args.early_stopping_patience:
                    print(f'Note: target can not be optimized for {args.early_stopping_patience} consecutive epochs, '
                        f'early stop the training process!')
                    break
            print('-' * 50)
        print('Best Score v: {:.4f}, a: {:.4f}\nBest Score a: {:.4f}, v: {:.4f}'.format(best_val_v, accp_val_a, best_val_a, accp_val_v))
        print('finishing training')


def train(model, train_loader, optimizer, loss_fn, eval_fn, device):
    model.train()
    loss_ = 0.0
    total = 0
    full_label = []
    full_preds = []
    for img, _,_,label,_ in train_loader:
        img = torch.cat(img, dim=1)
        img, label = img.to(device), label.to(device)
        batch_size = img.shape[0]
        optimizer.zero_grad()
        preds = model(img)
        # preds = preds.view(preds.size(0)*preds.size(1),-1).squeeze()
        # label = label.view(label.size(0)*label.size(1),-1).squeeze()
        loss = loss_fn(preds, label)
        loss.backward()
        optimizer.step()
        loss_ += loss.item() * batch_size
        total += batch_size
        full_label.append(label.cpu().detach().numpy())
        full_preds.append(preds.cpu().detach().numpy())
    full_preds = flatten_pred_label(full_preds)
    full_label = flatten_pred_label(full_label)
    score = eval_fn(full_preds, full_label)
    return loss_ / total, score

def evaluate(model, data_loader, loss_fn, eval_fn, device):
    model.eval()
    full_label = []
    full_preds = []
    with torch.no_grad():
        for img, _,_,label,_ in data_loader:
            img = torch.cat(img, dim=1)
            img, label = img.to(device), label.to(device)
            preds = model(img)
            # preds = preds.view(preds.size(0)*preds.size(1),-1).squeeze()
            # label = label.view(label.size(0)*label.size(1),-1).squeeze()
            full_label.append(label.cpu().detach().numpy())
            full_preds.append(preds.cpu().detach().numpy())
        full_preds = flatten_pred_label(full_preds)
        full_label = flatten_pred_label(full_label)
        loss = loss_fn(torch.from_numpy(full_preds), torch.from_numpy(full_label))
        score = eval_fn(full_preds, full_label)
    return loss, score

if __name__ == '__main__':
    args = parse_args()
    args.file_name = '{}_{}_{}'.format(datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M"), args.task, '+'.join(args.feature))
    os.makedirs('./checkpoint/'+args.file_name, exist_ok=True)
    sys.stdout = Logger('./checkpoint/' + args.file_name + '/log.txt')
    if args.task != 'VA':
        args.model_path = './checkpoint/' + args.file_name + '/best_model.pt'
    else:
        args.model_path = ['./checkpoint/' + args.file_name + '/best_v_model.pt', './checkpoint/' + args.file_name + '/best_a_model.pt']
    print(' '.join(sys.argv))
    print(args)
    main(args)