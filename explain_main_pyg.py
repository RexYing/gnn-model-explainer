from explain_configs import *
from models_pyg import GCNNet

def main():
  prog_args = arg_parse()
  path = os.path.join(prog_args.logdir, io_utils.gen_explainer_prefix(prog_args))
  writer = SummaryWriter(path)

  ckpt = io_utils.load_ckpt(prog_args)
  cg_dict = ckpt['cg']
  input_dim = cg_dict['feat'].shape[2]
  num_classes = cg_dict['pred'].shape[1]
  print('input dim: ', input_dim, '; num classes: ', num_classes)

  model = GCNNet(input_dim, prog_args.hidden_dim, prog_args.output_dim, num_classes, prog_args.num_gc_layers, args=prog_args)
  model.load_state_dict(ckpt['model_state'])
  train_idx = cg_dict['train_idx']

main()

