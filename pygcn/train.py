from __future__ import division
from __future__ import print_function
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='dblp',
                    help='dataset:dblp or imdb or freebase')
parser.add_argument('--metapath', type=str, default='APA',
                    help='dblp:APA,APCPA,APTPA;freebase:MAM MDM MWM;imdb:MAM MDM MKM')
parser.add_argument('--ratio', type=int, default=20)
parser.add_argument('--patience', type=int, default=30)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train_20, idx_val_20, idx_test_20, idx_train_40, idx_val_40, idx_test_40, idx_train_60, idx_val_60, idx_test_60 = load_data(args.dataset, args.metapath)

xent = torch.nn.CrossEntropyLoss()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)



def train(epoch):
    best_score = 0
    steps_now = 0
    for epoch_now in range(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        # print(output[idx_train].cpu().detach().numpy())
        # print(labels[idx_train].cpu().detach().numpy())
        loss_train = xent(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj)
        # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        output_soft = torch.softmax(output, dim = 1)

        output = torch.argmax(output, dim=1)
        # acc_val = accuracy(output[idx_val], labels[idx_val])
        macro = f1_score(labels[idx_val].cpu().detach().numpy(), output[idx_val].cpu().detach().numpy(), average='macro')
        micro = f1_score(labels[idx_val].cpu().detach().numpy(), output[idx_val].cpu().detach().numpy(), average='micro')

        auc = roc_auc_score(y_true=labels[idx_val].cpu().detach().numpy(), y_score=output_soft[idx_val].cpu().detach().numpy(), multi_class='ovr')
        score = auc + macro + micro
        if score > best_score:
            best_score = score
            steps_now = 0
            torch.save(model.state_dict(), 'GCN_'+ args.dataset + str(args.patience) +'.pkl')
        else:
            steps_now = steps_now + 1
            if steps_now == args.patience:
                # print('early stopping!')
                break

        # print('Epoch: {:04d}'.format(epoch_now+1),
        #     #   'loss_train: {:.4f}'.format(loss_train.item()),
        #     #   'acc_train: {:.4f}'.format(acc_train.item()),
        #     #   'loss_val: {:.4f}'.format(loss_val.item()),
        #     #   'acc_val: {:.4f}'.format(acc_val.item()),
        #     'step: {:.4f}'.format(steps_now),
        #     'time: {:.4f}s'.format(time.time() - t))
    # print('epoch:{}'.format(epoch_now))


def test():
    model.load_state_dict(torch.load('GCN_'+ args.dataset + str(args.patience) +'.pkl'))
    model.eval()
    output = model(features, adj)
    # # print(output.detach().numpy())
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    # acc_test = accuracy(output[idx_test], labels[idx_test])
    # print("Test set results:",
    #       "loss= {:.4f}".format(loss_test.item()),
    #       "accuracy= {:.4f}".format(acc_test.item()))
    # print(output.cpu().detach().numpy())
    output_soft = F.softmax(output, dim = 1).cpu().detach().numpy()
    output = torch.argmax(output, dim=1)
    output = output.cpu().detach().numpy()

    labels_c = labels.cpu().detach().numpy()
    idx_train_c = idx_train.cpu().detach().numpy()
    idx_val_c = idx_val.cpu().detach().numpy()
    idx_test_c = idx_test.cpu().detach().numpy()
    # print("output:{}, labels:{}".format(output, labels_c))
    auc = roc_auc_score(y_true=labels_c[idx_test_c], y_score=output_soft[idx_test_c], multi_class='ovr')
    f1_macro = f1_score(output[idx_test_c], labels_c[idx_test_c], average='macro')
    f1_micro = f1_score(output[idx_test_c], labels_c[idx_test_c], average='micro')
    print("\t[Classification] Macro-F1_mean: {:.4f}  Micro-F1_mean: {:.4f}  auc {:.4f}"
            .format(f1_macro, f1_micro, auc))

idx_train = idx_train_20
idx_val = idx_val_20
idx_test = idx_test_20
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
# Train model
t_total = time.time()
train(args.epochs)
# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

idx_train = idx_train_40
idx_val = idx_val_40
idx_test = idx_test_40
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
# Train model
t_total = time.time()
train(args.epochs)
# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

idx_train = idx_train_60
idx_val = idx_val_60
idx_test = idx_test_60
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
# Train model
t_total = time.time()
train(args.epochs)
# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
