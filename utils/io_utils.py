import os
import torch

def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
    name += '_' + args.method

    name += '_h' + str(args.hidden_dim) + '_o' + str(args.output_dim)
    if not args.bias:
        name += '_nobias'
    if len(args.name_suffix) > 0:
        name += '_' + args.name_suffix
    return name

def create_filename(args, isbest=False, num_epochs=-1):
    '''
    Args:
        args: the arguments parsed in the parser
        isbest: whether the saved model is the best-performing one
        num_epochs: epoch number of the model (when isbest=False)
    '''
    filename = os.path.join(args.ckptdir, gen_prefix(args))
    os.makedirs(filename, exist_ok=True)

    if isbest:
        filename = os.path.join(filename, 'best')
    elif num_epochs > 0:
        filename = os.path.join(filename, str(num_epochs))

    return filename + '.pth.tar'

def save_checkpoint(model, optimizer, args, num_epochs=-1, isbest=False):
    filename = create_filename(args, isbest, num_epochs=num_epochs)
    torch.save({'epoch': num_epochs,
                'model_type': args.method,
                'optimizer': optimizer,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()},
               filename)

def load_model(ckpt, model, model_opt):
    print('loading model')
    model.load_state_dict(ckpt['model_state'])
    model_opt.load_state_dict(ckpt['model_'])

