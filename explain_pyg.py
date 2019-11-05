import gengraph
import random
import torch_geometric
from utils import featgen
import numpy as np
import utils.io_utils as io_utils
from configs import arg_parse
import torch
import torch.nn as nn
from torch.autograd import Variable
from models_pyg import GCNNet
import os
from torch_geometric.utils import from_networkx
from tensorboardX import SummaryWriter

def test(loader, model, args, labels, test_mask):
    model.eval()

    train_ratio = args.train_ratio
    correct = 0
    for data in loader:
        with torch.no_grad():
            pred = model(data)
            # print ('pred:', pred)
            pred = pred.argmax(dim=1)
            # print ('pred:', pred)

        # node classification: only evaluate on nodes in test set
        pred = pred[test_mask]
        # print ('pred:', pred)
        label = labels[test_mask]
        # print ('label:', label)
            
        correct += pred.eq(label).sum().item()
    
    total = len(test_mask) 
    # print ('correct:', correct)
    return correct / total

def syn_task1(args, writer=None):
    # data
    print ('Generating graph.')
    G, labels, name = gengraph.gen_syn1(
            feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float)))
    # print ('G.node[0]:', G.node[0]['feat'].dtype)
    # print ('Original labels:', labels)
    pyg_G = from_networkx(G)
    num_classes = max(labels)+1
    labels = torch.LongTensor(labels)
    print ('Done generating graph.')

    # if args.method == 'att':
    # print('Method: att')
    # model = models.GcnEncoderNode(args.input_dim, args.hidden_dim, args.output_dim, num_classes,
    # args.num_gc_layers, bn=args.bn, args=args)

    # else:
    # print('Method:', args.method)
    # model = models.GcnEncoderNode(args.input_dim, args.hidden_dim, args.output_dim, num_classes,
    # args.num_gc_layers, bn=args.bn, args=args)

    model = GCNNet(args.input_dim, args.hidden_dim, num_classes, args.num_gc_layers, args=args)
    
    if args.gpu:
        model = model.cuda()

    train_ratio = args.train_ratio
    num_train = int(train_ratio * G.number_of_nodes())
    num_test = G.number_of_nodes() - num_train
    shuffle_indices = list(range(G.number_of_nodes()))
    shuffle_indices = np.random.permutation(shuffle_indices)

    train_mask = num_train * [True] + num_test * [False]
    train_mask = torch.BoolTensor([train_mask[i] for i in shuffle_indices])
    test_mask = num_train * [False] + num_test * [True]
    test_mask = torch.BoolTensor([test_mask[i] for i in shuffle_indices])

    loader = torch_geometric.data.DataLoader([pyg_G], batch_size=1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.num_epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            # print ('batch:', batch.feat)
            opt.zero_grad()
            pred = model(batch)
        
            pred = pred[train_mask]
            # print ('pred:', pred)
            label = labels[train_mask]
            # print ('label:', label)
            loss = model.loss(pred, label)
            print ('loss:', loss)
            loss.backward()
            opt.step()
            total_loss += loss.item() * 1
        total_loss /= num_train
        writer.add_scalar("loss", total_loss, epoch)
        
        if epoch % 10 == 0:
            test_acc = test(loader, model, args, labels, test_mask)
            print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(
                epoch, total_loss, test_acc))
            writer.add_scalar("test accuracy", test_acc, epoch)

prog_args = arg_parse()
path = os.path.join(prog_args.logdir, io_utils.gen_prefix(prog_args))
syn_task1(prog_args, writer=SummaryWriter(path))
