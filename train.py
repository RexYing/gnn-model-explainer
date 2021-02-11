""" train.py

    Main interface to train the GNNs that will be later explained.
"""
import argparse
import os
import pickle
import random
import shutil
import time

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import networkx as nx
import numpy as np
import sklearn.metrics as metrics

import torch
import torch.nn as nn
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import configs
import gengraph

import utils.math_utils as math_utils
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
import utils.train_utils as train_utils
import utils.featgen as featgen
import utils.graph_utils as graph_utils

import models


#############################
#
# Prepare Data
#
#############################
def prepare_data(graphs, args, test_graphs=None, max_nodes=0):

    random.shuffle(graphs)
    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1 - args.test_ratio))
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx:test_idx]
        test_graphs = graphs[test_idx:]
    else:
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = graphs[:train_idx]
        val_graphs = graph[train_idx:]
    print(
        "Num training graphs: ",
        len(train_graphs),
        "; Num validation graphs: ",
        len(val_graphs),
        "; Num testing graphs: ",
        len(test_graphs),
    )

    print("Number of graphs: ", len(graphs))
    print("Number of edges: ", sum([G.number_of_edges() for G in graphs]))
    print(
        "Max, avg, std of graph size: ",
        max([G.number_of_nodes() for G in graphs]),
        ", " "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])),
        ", " "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])),
    )

    # minibatch
    dataset_sampler = graph_utils.GraphSampler(
        train_graphs,
        normalize=False,
        max_num_nodes=max_nodes,
        features=args.feature_type,
    )
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    dataset_sampler = graph_utils.GraphSampler(
        val_graphs, 
        normalize=False, 
        max_num_nodes=max_nodes, 
        features=args.feature_type
    )
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    dataset_sampler = graph_utils.GraphSampler(
        test_graphs,
        normalize=False,
        max_num_nodes=max_nodes,
        features=args.feature_type,
    )
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return (
        train_dataset_loader,
        val_dataset_loader,
        test_dataset_loader,
        dataset_sampler.max_num_nodes,
        dataset_sampler.feat_dim,
        dataset_sampler.assign_feat_dim,
    )


#############################
#
# Training 
#
#############################
def train(
    dataset,
    model,
    args,
    same_feat=True,
    val_dataset=None,
    test_dataset=None,
    writer=None,
    mask_nodes=True,
):
    writer_batch_idx = [0, 3, 6, 9]

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )
    iter = 0
    best_val_result = {"epoch": 0, "loss": 0, "acc": 0}
    test_result = {"epoch": 0, "loss": 0, "acc": 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []

    for epoch in range(args.num_epochs):
        begin_time = time.time()
        avg_loss = 0.0
        model.train()
        predictions = []
        print("Epoch: ", epoch)
        for batch_idx, data in enumerate(dataset):
            model.zero_grad()
            if batch_idx == 0:
                prev_adjs = data["adj"]
                prev_feats = data["feats"]
                prev_labels = data["label"]
                all_adjs = prev_adjs
                all_feats = prev_feats
                all_labels = prev_labels
            elif batch_idx < 20:
                prev_adjs = data["adj"]
                prev_feats = data["feats"]
                prev_labels = data["label"]
                all_adjs = torch.cat((all_adjs, prev_adjs), dim=0)
                all_feats = torch.cat((all_feats, prev_feats), dim=0)
                all_labels = torch.cat((all_labels, prev_labels), dim=0)
            adj = Variable(data["adj"].float(), requires_grad=False).cuda()
            h0 = Variable(data["feats"].float(), requires_grad=False).cuda()
            label = Variable(data["label"].long()).cuda()
            batch_num_nodes = data["num_nodes"].int().numpy() if mask_nodes else None
            assign_input = Variable(
                data["assign_feats"].float(), requires_grad=False
            ).cuda()

            ypred, att_adj = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            if batch_idx < 5:
                predictions += ypred.cpu().detach().numpy().tolist()

            if not args.method == "soft-assign" or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss

        avg_loss /= batch_idx + 1
        elapsed = time.time() - begin_time
        if writer is not None:
            writer.add_scalar("loss/avg_loss", avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar("loss/linkpred_loss", model.link_loss, epoch)
        print("Avg loss: ", avg_loss, "; epoch time: ", elapsed)
        result = evaluate(dataset, model, args, name="Train", max_num_examples=100)
        train_accs.append(result["acc"])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, args, name="Validation")
            val_accs.append(val_result["acc"])
        if val_result["acc"] > best_val_result["acc"] - 1e-7:
            best_val_result["acc"] = val_result["acc"]
            best_val_result["epoch"] = epoch
            best_val_result["loss"] = avg_loss
        if test_dataset is not None:
            test_result = evaluate(test_dataset, model, args, name="Test")
            test_result["epoch"] = epoch
        if writer is not None:
            writer.add_scalar("acc/train_acc", result["acc"], epoch)
            writer.add_scalar("acc/val_acc", val_result["acc"], epoch)
            writer.add_scalar("loss/best_val_loss", best_val_result["loss"], epoch)
            if test_dataset is not None:
                writer.add_scalar("acc/test_acc", test_result["acc"], epoch)

        print("Best val result: ", best_val_result)
        best_val_epochs.append(best_val_result["epoch"])
        best_val_accs.append(best_val_result["acc"])
        if test_dataset is not None:
            print("Test result: ", test_result)
            test_epochs.append(test_result["epoch"])
            test_accs.append(test_result["acc"])

    matplotlib.style.use("seaborn")
    plt.switch_backend("agg")
    plt.figure()
    plt.plot(train_epochs, math_utils.exp_moving_avg(train_accs, 0.85), "-", lw=1)
    if test_dataset is not None:
        plt.plot(best_val_epochs, best_val_accs, "bo", test_epochs, test_accs, "go")
        plt.legend(["train", "val", "test"])
    else:
        plt.plot(best_val_epochs, best_val_accs, "bo")
        plt.legend(["train", "val"])
    plt.savefig(io_utils.gen_train_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use("default")

    print(all_adjs.shape, all_feats.shape, all_labels.shape)

    cg_data = {
        "adj": all_adjs,
        "feat": all_feats,
        "label": all_labels,
        "pred": np.expand_dims(predictions, axis=0),
        "train_idx": list(range(len(dataset))),
    }
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)
    return model, val_accs


def train_node_classifier(G, labels, model, args, writer=None):
    # train/test split only for nodes
    num_nodes = G.number_of_nodes()
    num_train = int(num_nodes * args.train_ratio)
    idx = [i for i in range(num_nodes)]

    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]

    data = gengraph.preprocess_input_graph(G, labels)
    labels_train = torch.tensor(data["labels"][:, train_idx], dtype=torch.long)
    adj = torch.tensor(data["adj"], dtype=torch.float)
    x = torch.tensor(data["feat"], requires_grad=True, dtype=torch.float)
    scheduler, optimizer = train_utils.build_optimizer(
        args, model.parameters(), weight_decay=args.weight_decay
    )
    model.train()
    ypred = None
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        model.zero_grad()

        if args.gpu:
            ypred, adj_att = model(x.cuda(), adj.cuda())
        else:
            ypred, adj_att = model(x, adj)
        ypred_train = ypred[:, train_idx, :]
        if args.gpu:
            loss = model.loss(ypred_train, labels_train.cuda())
        else:
            loss = model.loss(ypred_train, labels_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        #for param_group in optimizer.param_groups:
        #    print(param_group["lr"])
        elapsed = time.time() - begin_time

        result_train, result_test = evaluate_node(
            ypred.cpu(), data["labels"], train_idx, test_idx
        )
        if writer is not None:
            writer.add_scalar("loss/avg_loss", loss, epoch)
            writer.add_scalars(
                "prec",
                {"train": result_train["prec"], "test": result_test["prec"]},
                epoch,
            )
            writer.add_scalars(
                "recall",
                {"train": result_train["recall"], "test": result_test["recall"]},
                epoch,
            )
            writer.add_scalars(
                "acc", {"train": result_train["acc"], "test": result_test["acc"]}, epoch
            )

        if epoch % 10 == 0:
            print(
                "epoch: ",
                epoch,
                "; loss: ",
                loss.item(),
                "; train_acc: ",
                result_train["acc"],
                "; test_acc: ",
                result_test["acc"],
                "; train_prec: ",
                result_train["prec"],
                "; test_prec: ",
                result_test["prec"],
                "; epoch time: ",
                "{0:0.2f}".format(elapsed),
            )

        if scheduler is not None:
            scheduler.step()
    print(result_train["conf_mat"])
    print(result_test["conf_mat"])

    # computation graph
    model.eval()
    if args.gpu:
        ypred, _ = model(x.cuda(), adj.cuda())
    else:
        ypred, _ = model(x, adj)
    cg_data = {
        "adj": data["adj"],
        "feat": data["feat"],
        "label": data["labels"],
        "pred": ypred.cpu().detach().numpy(),
        "train_idx": train_idx,
    }
    # import pdb
    # pdb.set_trace()
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)


def train_node_classifier_multigraph(G_list, labels, model, args, writer=None):
    train_idx_all, test_idx_all = [], []
    # train/test split only for nodes
    num_nodes = G_list[0].number_of_nodes()
    num_train = int(num_nodes * args.train_ratio)
    idx = [i for i in range(num_nodes)]
    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    train_idx_all.append(train_idx)
    test_idx = idx[num_train:]
    test_idx_all.append(test_idx)

    data = gengraph.preprocess_input_graph(G_list[0], labels[0])
    all_labels = data["labels"]
    labels_train = torch.tensor(data["labels"][:, train_idx], dtype=torch.long)
    adj = torch.tensor(data["adj"], dtype=torch.float)
    x = torch.tensor(data["feat"], requires_grad=True, dtype=torch.float)

    for i in range(1, len(G_list)):
        np.random.shuffle(idx)
        train_idx = idx[:num_train]
        train_idx_all.append(train_idx)
        test_idx = idx[num_train:]
        test_idx_all.append(test_idx)
        data = gengraph.preprocess_input_graph(G_list[i], labels[i])
        all_labels = np.concatenate((all_labels, data["labels"]), axis=0)
        labels_train = torch.cat(
            [
                labels_train,
                torch.tensor(data["labels"][:, train_idx], dtype=torch.long),
            ],
            dim=0,
        )
        adj = torch.cat([adj, torch.tensor(data["adj"], dtype=torch.float)])
        x = torch.cat(
            [x, torch.tensor(data["feat"], requires_grad=True, dtype=torch.float)]
        )

    scheduler, optimizer = train_utils.build_optimizer(
        args, model.parameters(), weight_decay=args.weight_decay
    )
    model.train()
    ypred = None
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        model.zero_grad()

        if args.gpu:
            ypred = model(x.cuda(), adj.cuda())
        else:
            ypred = model(x, adj)
        # normal indexing
        ypred_train = ypred[:, train_idx, :]
        # in multigraph setting we can't directly access all dimensions so we need to gather all the training instances
        all_train_idx = [item for sublist in train_idx_all for item in sublist]
        ypred_train_cmp = torch.cat(
            [ypred[i, train_idx_all[i], :] for i in range(10)], dim=0
        ).reshape(10, 146, 6)
        if args.gpu:
            loss = model.loss(ypred_train_cmp, labels_train.cuda())
        else:
            loss = model.loss(ypred_train_cmp, labels_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        #for param_group in optimizer.param_groups:
        #    print(param_group["lr"])
        elapsed = time.time() - begin_time

        result_train, result_test = evaluate_node(
            ypred.cpu(), all_labels, train_idx_all, test_idx_all
        )
        if writer is not None:
            writer.add_scalar("loss/avg_loss", loss, epoch)
            writer.add_scalars(
                "prec",
                {"train": result_train["prec"], "test": result_test["prec"]},
                epoch,
            )
            writer.add_scalars(
                "recall",
                {"train": result_train["recall"], "test": result_test["recall"]},
                epoch,
            )
            writer.add_scalars(
                "acc", {"train": result_train["acc"], "test": result_test["acc"]}, epoch
            )

        print(
            "epoch: ",
            epoch,
            "; loss: ",
            loss.item(),
            "; train_acc: ",
            result_train["acc"],
            "; test_acc: ",
            result_test["acc"],
            "; epoch time: ",
            "{0:0.2f}".format(elapsed),
        )

        if scheduler is not None:
            scheduler.step()
    print(result_train["conf_mat"])
    print(result_test["conf_mat"])

    # computation graph
    model.eval()
    if args.gpu:
        ypred = model(x.cuda(), adj.cuda())
    else:
        ypred = model(x, adj)
    cg_data = {
        "adj": adj.cpu().detach().numpy(),
        "feat": x.cpu().detach().numpy(),
        "label": all_labels,
        "pred": ypred.cpu().detach().numpy(),
        "train_idx": train_idx_all,
    }
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)



#############################
#
# Evaluate Trained Model
#
#############################
def evaluate(dataset, model, args, name="Validation", max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data["adj"].float(), requires_grad=False).cuda()
        h0 = Variable(data["feats"].float()).cuda()
        labels.append(data["label"].long().numpy())
        batch_num_nodes = data["num_nodes"].int().numpy()
        assign_input = Variable(
            data["assign_feats"].float(), requires_grad=False
        ).cuda()

        ypred, att_adj = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    result = {
        "prec": metrics.precision_score(labels, preds, average="macro"),
        "recall": metrics.recall_score(labels, preds, average="macro"),
        "acc": metrics.accuracy_score(labels, preds),
    }
    print(name, " accuracy:", result["acc"])
    return result


def evaluate_node(ypred, labels, train_idx, test_idx):
    _, pred_labels = torch.max(ypred, 2)
    pred_labels = pred_labels.numpy()

    pred_train = np.ravel(pred_labels[:, train_idx])
    pred_test = np.ravel(pred_labels[:, test_idx])
    labels_train = np.ravel(labels[:, train_idx])
    labels_test = np.ravel(labels[:, test_idx])

    result_train = {
        "prec": metrics.precision_score(labels_train, pred_train, average="macro"),
        "recall": metrics.recall_score(labels_train, pred_train, average="macro"),
        "acc": metrics.accuracy_score(labels_train, pred_train),
        "conf_mat": metrics.confusion_matrix(labels_train, pred_train),
    }
    result_test = {
        "prec": metrics.precision_score(labels_test, pred_test, average="macro"),
        "recall": metrics.recall_score(labels_test, pred_test, average="macro"),
        "acc": metrics.accuracy_score(labels_test, pred_test),
        "conf_mat": metrics.confusion_matrix(labels_test, pred_test),
    }
    return result_train, result_test



#############################
#
# Run Experiments
#
#############################
def ppi_essential_task(args, writer=None):
    feat_file = "G-MtfPathways_gene-motifs.csv"
    # G = io_utils.read_biosnap('data/ppi_essential', 'PP-Pathways_ppi.csv', 'G-HumanEssential.tsv',
    #        feat_file=feat_file)
    G = io_utils.read_biosnap(
        "data/ppi_essential",
        "hi-union-ppi.tsv",
        "G-HumanEssential.tsv",
        feat_file=feat_file,
    )
    labels = np.array([G.nodes[u]["label"] for u in G.nodes()])
    num_classes = max(labels) + 1
    input_dim = G.nodes[next(iter(G.nodes()))]["feat"].shape[0]

    if args.method == "attn":
        print("Method: attn")
    else:
        print("Method:", args.method)
        args.loss_weight = torch.tensor([1, 5.0], dtype=torch.float).cuda()
        model = models.GcnEncoderNode(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
        if args.gpu:
            model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)


def syn_task1(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn1(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
    num_classes = max(labels) + 1

    if args.method == "att":
        print("Method: att")
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
    else:
        print("Method:", args.method)
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
    if args.gpu:
        model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)


def syn_task2(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn2()
    input_dim = len(G.nodes[0]["feat"])
    num_classes = max(labels) + 1

    if args.method == "attn":
        print("Method: attn")
    else:
        print("Method:", args.method)
        model = models.GcnEncoderNode(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
        if args.gpu:
            model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)


def syn_task3(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn3(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
    print(labels)
    num_classes = max(labels) + 1

    if args.method == "attn":
        print("Method: attn")
    else:
        print("Method:", args.method)
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
        if args.gpu:
            model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)


def syn_task4(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn4(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
    print(labels)
    num_classes = max(labels) + 1

    if args.method == "attn":
        print("Method: attn")
    else:
        print("Method:", args.method)
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )

        if args.gpu:
            model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)


def syn_task5(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn5(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
    print(labels)
    print("Number of nodes: ", G.number_of_nodes())
    num_classes = max(labels) + 1

    if args.method == "attn":
        print("Method: attn")
    else:
        print("Method: base")
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )

        if args.gpu:
            model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)


def pkl_task(args, feat=None):
    with open(os.path.join(args.datadir, args.pkl_fname), "rb") as pkl_file:
        data = pickle.load(pkl_file)
    graphs = data[0]
    labels = data[1]
    test_graphs = data[2]
    test_labels = data[3]

    for i in range(len(graphs)):
        graphs[i].graph["label"] = labels[i]
    for i in range(len(test_graphs)):
        test_graphs[i].graph["label"] = test_labels[i]

    if feat is None:
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)
        for G in test_graphs:
            featgen_const.gen_node_features(G)

    train_dataset, test_dataset, max_num_nodes = prepare_data(
        graphs, args, test_graphs=test_graphs
    )
    model = models.GcnEncoderGraph(
        args.input_dim,
        args.hidden_dim,
        args.output_dim,
        args.num_classes,
        args.num_gc_layers,
        bn=args.bn,
    ).cuda()
    train(train_dataset, model, args, test_dataset=test_dataset)
    evaluate(test_dataset, model, args, "Validation")


def enron_task_multigraph(args, idx=None, writer=None):
    labels_dict = {
        "None": 5,
        "Employee": 0,
        "Vice President": 1,
        "Manager": 2,
        "Trader": 3,
        "CEO+Managing Director+Director+President": 4,
    }
    max_enron_id = 183
    if idx is None:
        G_list = []
        labels_list = []
        for i in range(10):
            net = pickle.load(
                open("data/gnn-explainer-enron/enron_slice_{}.pkl".format(i), "rb")
            )
            net.add_nodes_from(range(max_enron_id))
            labels = [n[1].get("role", "None") for n in net.nodes(data=True)]
            labels_num = [labels_dict[l] for l in labels]
            featgen_const = featgen.ConstFeatureGen(
                np.ones(args.input_dim, dtype=float)
            )
            featgen_const.gen_node_features(net)
            G_list.append(net)
            labels_list.append(labels_num)
        # train_dataset, test_dataset, max_num_nodes = prepare_data(G_list, args)
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
        if args.gpu:
            model = model.cuda()
        print(labels_num)
        train_node_classifier_multigraph(
            G_list, labels_list, model, args, writer=writer
        )
    else:
        print("Running Enron full task")


def enron_task(args, idx=None, writer=None):
    labels_dict = {
        "None": 5,
        "Employee": 0,
        "Vice President": 1,
        "Manager": 2,
        "Trader": 3,
        "CEO+Managing Director+Director+President": 4,
    }
    max_enron_id = 183
    if idx is None:
        G_list = []
        labels_list = []
        for i in range(10):
            net = pickle.load(
                open("data/gnn-explainer-enron/enron_slice_{}.pkl".format(i), "rb")
            )
            # net.add_nodes_from(range(max_enron_id))
            # labels=[n[1].get('role', 'None') for n in net.nodes(data=True)]
            # labels_num = [labels_dict[l] for l in labels]
            featgen_const = featgen.ConstFeatureGen(
                np.ones(args.input_dim, dtype=float)
            )
            featgen_const.gen_node_features(net)
            G_list.append(net)
            print(net.number_of_nodes())
            # labels_list.append(labels_num)

        G = nx.disjoint_union_all(G_list)
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            len(labels_dict),
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
        labels = [n[1].get("role", "None") for n in G.nodes(data=True)]
        labels_num = [labels_dict[l] for l in labels]
        for i in range(5):
            print("Label ", i, ": ", labels_num.count(i))

        print("Total num nodes: ", len(labels_num))
        print(labels_num)

        if args.gpu:
            model = model.cuda()
        train_node_classifier(G, labels_num, model, args, writer=writer)
    else:
        print("Running Enron full task")


def benchmark_task(args, writer=None, feat="node-label"):
    graphs = io_utils.read_graphfile(
        args.datadir, args.bmname, max_nodes=args.max_nodes
    )
    print(max([G.graph["label"] for G in graphs]))

    if feat == "node-feat" and "feat_dim" in graphs[0].graph:
        print("Using node features")
        input_dim = graphs[0].graph["feat_dim"]
    elif feat == "node-label" and "label" in graphs[0].nodes[0]:
        print("Using node labels")
        for G in graphs:
            for u in G.nodes():
                G.nodes[u]["feat"] = np.array(G.nodes[u]["label"])
                # make it -1/1 instead of 0/1
                # feat = np.array(G.nodes[u]['label'])
                # G.nodes[u]['feat'] = feat * 2 - 1
    else:
        print("Using constant labels")
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_data(
        graphs, args, max_nodes=args.max_nodes
    )
    if args.method == "soft-assign":
        print("Method: soft-assign")
        model = models.SoftPoolingGcnEncoder(
            max_num_nodes,
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            args.hidden_dim,
            assign_ratio=args.assign_ratio,
            num_pooling=args.num_pool,
            bn=args.bn,
            dropout=args.dropout,
            linkpred=args.linkpred,
            args=args,
            assign_input_dim=assign_input_dim,
        ).cuda()
    else:
        print("Method: base")
        model = models.GcnEncoderGraph(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            bn=args.bn,
            dropout=args.dropout,
            args=args,
        ).cuda()

    train(
        train_dataset,
        model,
        args,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        writer=writer,
    )
    evaluate(test_dataset, model, args, "Validation")


def benchmark_task_val(args, writer=None, feat="node-label"):
    all_vals = []
    graphs = io_utils.read_graphfile(
        args.datadir, args.bmname, max_nodes=args.max_nodes
    )

    if feat == "node-feat" and "feat_dim" in graphs[0].graph:
        print("Using node features")
        input_dim = graphs[0].graph["feat_dim"]
    elif feat == "node-label" and "label" in graphs[0].nodes[0]:
        print("Using node labels")
        for G in graphs:
            for u in G.nodes():
                G.nodes[u]["feat"] = np.array(G.nodes[u]["label"])
    else:
        print("Using constant labels")
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    # 10 splits
    for i in range(10):
        train_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = cross_val.prepare_val_data(
            graphs, args, i, max_nodes=args.max_nodes
        )
        print("Method: base")
        model = models.GcnEncoderGraph(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            bn=args.bn,
            dropout=args.dropout,
            args=args,
        ).cuda()

        _, val_accs = train(
            train_dataset,
            model,
            args,
            val_dataset=val_dataset,
            test_dataset=None,
            writer=writer,
        )
        all_vals.append(np.array(val_accs))
    all_vals = np.vstack(all_vals)
    all_vals = np.mean(all_vals, axis=0)
    print(all_vals)
    print(np.max(all_vals))
    print(np.argmax(all_vals))


def arg_parse():
    parser = argparse.ArgumentParser(description="GraphPool arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument(
        "--bmname", dest="bmname", help="Name of the benchmark dataset"
    )
    io_parser.add_argument("--pkl", dest="pkl_fname", help="Name of the pkl data file")

    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument(
        "--assign-ratio",
        dest="assign_ratio",
        type=float,
        help="ratio of number of nodes in consecutive layers",
    )
    softpool_parser.add_argument(
        "--num-pool", dest="num_pool", type=int, help="number of pooling layers"
    )
    parser.add_argument(
        "--linkpred",
        dest="linkpred",
        action="store_const",
        const=True,
        default=False,
        help="Whether link prediction side objective is used",
    )

    parser_utils.parse_optimizer(parser)

    parser.add_argument(
        "--datadir", dest="datadir", help="Directory where benchmark is located"
    )
    parser.add_argument("--logdir", dest="logdir", help="Tensorboard log directory")
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory")
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_const",
        const=True,
        default=False,
        help="whether to use GPU.",
    )
    parser.add_argument(
        "--max-nodes",
        dest="max_nodes",
        type=int,
        help="Maximum number of nodes (ignore graghs with nodes exceeding the number.",
    )
    parser.add_argument("--batch-size", dest="batch_size", type=int, help="Batch size.")
    parser.add_argument(
        "--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--train-ratio",
        dest="train_ratio",
        type=float,
        help="Ratio of number of graphs training set to all graphs.",
    )
    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        type=int,
        help="Number of workers to load data.",
    )
    parser.add_argument(
        "--feature",
        dest="feature_type",
        help="Feature used for encoder. Can be: id, deg",
    )
    parser.add_argument(
        "--input-dim", dest="input_dim", type=int, help="Input feature dimension"
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num-classes", dest="num_classes", type=int, help="Number of label classes"
    )
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        type=int,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=False,
        help="Whether batch normalization is used",
    )
    parser.add_argument("--dropout", dest="dropout", type=float, help="Dropout rate.")
    parser.add_argument(
        "--nobias",
        dest="bias",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        help="Weight decay regularization constant.",
    )

    parser.add_argument(
        "--method", dest="method", help="Method. Possible values: base, "
    )
    parser.add_argument(
        "--name-suffix", dest="name_suffix", help="suffix added to the output filename"
    )

    parser.set_defaults(
        datadir="data",  # io_parser
        logdir="log",
        ckptdir="ckpt",
        dataset="syn1",
        opt="adam",  # opt_parser
        opt_scheduler="none",
        max_nodes=100,
        cuda="1",
        feature_type="default",
        lr=0.001,
        clip=2.0,
        batch_size=20,
        num_epochs=1000,
        train_ratio=0.8,
        test_ratio=0.1,
        num_workers=1,
        input_dim=10,
        hidden_dim=20,
        output_dim=20,
        num_classes=2,
        num_gc_layers=3,
        dropout=0.0,
        weight_decay=0.005,
        method="base",
        name_suffix="",
        assign_ratio=0.1,
    )
    return parser.parse_args()


def main():
    prog_args = configs.arg_parse()

    path = os.path.join(prog_args.logdir, io_utils.gen_prefix(prog_args))
    writer = SummaryWriter(path)

    if prog_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        print("CUDA", prog_args.cuda)
    else:
        print("Using CPU")

    # use --bmname=[dataset_name] for Reddit-Binary, Mutagenicity
    if prog_args.bmname is not None:
        benchmark_task(prog_args, writer=writer)
    elif prog_args.pkl_fname is not None:
        pkl_task(prog_args)
    elif prog_args.dataset is not None:
        if prog_args.dataset == "syn1":
            syn_task1(prog_args, writer=writer)
        elif prog_args.dataset == "syn2":
            syn_task2(prog_args, writer=writer)
        elif prog_args.dataset == "syn3":
            syn_task3(prog_args, writer=writer)
        elif prog_args.dataset == "syn4":
            syn_task4(prog_args, writer=writer)
        elif prog_args.dataset == "syn5":
            syn_task5(prog_args, writer=writer)
        elif prog_args.dataset == "enron":
            enron_task(prog_args, writer=writer)
        elif prog_args.dataset == "ppi_essential":
            ppi_essential_task(prog_args, writer=writer)

    writer.close()


if __name__ == "__main__":
    main()

