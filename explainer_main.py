
import sklearn.metrics as metrics
from tensorboardX import SummaryWriter


import argparse
import os
import pickle
import shutil

import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
import models
from explainer import  explain


def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset',
            help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname',
            help='Name of the benchmark dataset')
    io_parser.add_argument('--pkl', dest='pkl_fname',
            help='Name of the pkl data file')

    parser_utils.parse_optimizer(parser)

    parser.add_argument('--logdir', dest='logdir',
            help='Tensorboard log directory')
    parser.add_argument('--ckptdir', dest='ckptdir',
            help='Model checkpoint directory')
    parser.add_argument('--cuda', dest='cuda',
            help='CUDA.')
    parser.add_argument('--gpu', dest='gpu', action='store_const',
            const=True, default=False,
            help='whether to use GPU.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
            help='Number of epochs to train.')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
            help='Output dimension')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--bn', dest='bn', action='store_const',
            const=True, default=False,
            help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
            const=False, default=True,
            help='Whether to add bias. Default to True.')
    parser.add_argument('--no-writer', dest='writer', action='store_const',
            const=False, default=True,
            help='Whether to add bias. Default to True.')
    # Explainer
    parser.add_argument('--mask-bias', dest='mask_bias', action='store_const',
            const=True, default=False,
            help='Whether to add bias. Default to True.')
    parser.add_argument('--explain-node', dest='explain_node', type=int,
            help='Node to explain.')
    parser.add_argument('--align_steps', dest='align_steps', type=int,
            help='Number of iterations to find P, the alignment matrix.')

    parser.add_argument('--method', dest='method',
            help='Method. Possible values: base, ')
    parser.add_argument('--name-suffix', dest='name_suffix',
            help='suffix added to the output filename')

    parser.set_defaults(logdir='log',
                        ckptdir='ckpt',
                        dataset='syn1',
                        opt='adam',   # opt_parser
                        opt_scheduler='none',
                        cuda='0',
                        lr=0.1,
                        clip=2.0,
                        batch_size=20,
                        num_epochs=1000,
                        hidden_dim=20,
                        output_dim=20,
                        num_gc_layers=3,
                        dropout=0.0,
                        method='base',
                        name_suffix='',
                        align_steps=1000,
                        explain_node=420
                       )
    return parser.parse_args()

def main():
    prog_args = arg_parse()

    if prog_args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda
        print('CUDA', prog_args.cuda)
    else:
        print('Using CPU')

    if prog_args.writer:
        path = os.path.join(prog_args.logdir, io_utils.gen_explainer_prefix(prog_args))
        #if os.path.isdir(path):
        #    print('Remove existing log dir: ', path)
        #    shutil.rmtree(path)
        writer = SummaryWriter(path)
    else:
        writer = None

    ckpt = io_utils.load_ckpt(prog_args)
    cg_dict = ckpt['cg']
    input_dim = cg_dict['feat'].shape[2]
    num_classes = cg_dict['pred'].shape[2]
    print('input dim: ', input_dim, '; num classes: ', num_classes)

    # build model
    if prog_args.method == 'attn':
        print('Method: attn')
    else:
        print('Method: base')
        model = models.GcnEncoderNode(input_dim, prog_args.hidden_dim, prog_args.output_dim, num_classes,
                                       prog_args.num_gc_layers, bn=prog_args.bn, args=prog_args)
        if prog_args.gpu:
            model = model.cuda()
        model.load_state_dict(ckpt['model_state'])

        explainer = explain.Explainer(model, cg_dict['adj'], cg_dict['feat'],
                                      cg_dict['label'], cg_dict['pred'], cg_dict['train_idx'],
                                      prog_args, writer=writer, print_training=True)
        train_idx = cg_dict['train_idx']
#        explainer.explain(prog_args.explain_node, unconstrained=False)

        # explain a set of nodes
        masked_adj = explainer.explain_nodes([420,440], prog_args)
        #pickle.dump(masked_adj, open('out/masked_adjs.pkl', 'wb'))

        #print(masked_adj[3])

        
if __name__ == "__main__":
    main()

