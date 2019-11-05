from tensorboardX import SummaryWriter
import argparse
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
import os

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
    parser.add_argument('--mask-act', dest='mask_act', type=str,
            help='sigmoid, ReLU.')
    parser.add_argument('--mask-bias', dest='mask_bias', action='store_const',
            const=True, default=False,
            help='Whether to add bias. Default to True.')
    parser.add_argument('--explain-node', dest='explain_node', type=int,
            help='Node to explain.')
    parser.add_argument('--graph-idx', dest='graph_idx', type=int,
            help='Graph to explain.')
    parser.add_argument('--graph-mode', dest='graph_mode', action='store_const',
            const=True, default=False,
            help='whether to run Explainer on Graph Classification task.')
    parser.add_argument('--multigraph-class', dest='multigraph_class', type=int,
            help='whether to run Explainer on multiple Graphs from the Classification task for examples in the same class.')
    parser.add_argument('--multinode-class', dest='multinode_class', type=int,
            help='whether to run Explainer on multiple nodes from the Classification task for examples in the same class.')
    parser.add_argument('--align-steps', dest='align_steps', type=int,
            help='Number of iterations to find P, the alignment matrix.')

    parser.add_argument('--method', dest='method', type=str,
            help='Method. Possible values: base, att')
    parser.add_argument('--name-suffix', dest='name_suffix',
            help='suffix added to the output filename')
    parser.add_argument('--explainer-suffix', dest='explainer_suffix',
            help='suffix added to the explainer log')

    parser.set_defaults(logdir='log',
                        ckptdir='ckpt',
                        dataset='syn1',
                        opt='adam',   # opt_parser
                        opt_scheduler='none',
                        cuda='0',
                        lr=0.1,
                        clip=2.0,
                        batch_size=20,
                        num_epochs=500,
                        hidden_dim=20,
                        output_dim=20,
                        num_gc_layers=3,
                        dropout=0.0,
                        method='base',
                        name_suffix='',
                        explainer_suffix='',
                        align_steps=1000,
                        explain_node=None,
                        graph_idx=-1,
                        mask_act='sigmoid',
                        multigraph_class=-1,
                        multinode_class=-1
                       )
    return parser.parse_args()

