import torch
import os, sys, platform

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies import DDPStrategy

if int(pl.__version__.split('.')[0]) >= 2:
    from pytorch_lightning.plugins import MixedPrecisionPlugin
else:
    from pytorch_lightning.plugins import NativeMixedPrecisionPlugin as MixedPrecisionPlugin

sys.path.append(os.path.dirname(os.path.dirname(sys.argv[0])))

from drughive.lightning import HVAEComplexSplit
from drughive.trainutils import Hparams, get_checkpoints_from_dir

import warnings
warnings.filterwarnings("ignore", message=".*The epoch parameter in .* was not necessary")
warnings.filterwarnings("ignore", message=".*lr_scheduler.step()")
warnings.filterwarnings("ignore", message=".*Warning: Samples outside of range")


def main(args):
    torch.autograd.set_detect_anomaly(True)

    seed_everything(42, workers=True) # reproducibility. also set deterministic=True in Trainer

    callbacks=[]

    checkpoint_periodic = ModelCheckpoint(dirpath=None, 
                                          every_n_epochs=args.checkpoint_every_n_epochs,
                                          save_top_k=args.checkpoint_save_top_k,
                                          monitor='epoch',
                                          mode='max',
                                          save_weights_only=args.checkpoint_weights_only)
    callbacks.append(checkpoint_periodic)

    
    checkpoint_latest = ModelCheckpoint(dirpath=None, 
                                          every_n_epochs=1,
                                          save_top_k=1,
                                          monitor='epoch',
                                          mode='max',
                                          save_weights_only=False)
    callbacks.append(checkpoint_latest)


    plugins = []

    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    precision_plugin = MixedPrecisionPlugin(precision=16, device='cuda', scaler=grad_scaler)
    plugins.append(precision_plugin)

    if args.num_devices > 1:
        plugins.append(pl.plugins.TorchSyncBatchNorm())


    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.save,
                                             name=args.run_name)

    
    ddp = DDPStrategy(process_group_backend='gloo' if platform.system() == 'Windows' else 'nccl', 
                      find_unused_parameters=False)

    epochs_max = args.epochs
    if hasattr(args, 'epochs_early_stop'):
        epochs_max = min(args.epochs_early_stop, epochs_max)

    trainer = Trainer(
        deterministic=False,
        devices=args.num_devices, 
        accelerator="gpu",
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val, 
        gradient_clip_algorithm=args.clip_gradients,
        strategy=ddp,
        logger=tb_logger,
        log_every_n_steps=100,
        max_epochs=args.epochs,
        plugins=plugins,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        val_check_interval=500,
        check_val_every_n_epoch=None,
        )

    model_class = eval(args.dict.get('model_class', 'HVAE'))

    if args.continue_from_checkpoint is not None:
        if args.continue_from_checkpoint.endswith('.ckpt'):
            ckpt_path = args.continue_from_checkpoint
        else:
            ckpt_files = get_checkpoints_from_dir(args.continue_from_checkpoint)
            ckpt_path = ckpt_files[-1]

        try:
            model = model_class.load_from_checkpoint(ckpt_path, args=args)
        except Exception as e:
            print('ckpt_path:', ckpt_path)
            print('args:', args)
            raise e
        model.args.continue_from_checkpoint = args.continue_from_checkpoint

        checkpoint = torch.load(ckpt_path, map_location='cpu')

        # load state dict for precision plugin
        mpkey = [x for x in checkpoint.keys() if 'MixedPrecisionPlugin' in x]
        if len(mpkey) > 0:
            print('Loading state dict for precision plugin')
            trainer.precision_plugin.load_state_dict(checkpoint[mpkey[0]])

        
        trainer.resume_epoch = checkpoint['epoch']
        trainer.resume_global_step = checkpoint['global_step']
        trainer.fit(model, ckpt_path=ckpt_path)
    else:
        model = model_class(args)
        trainer.fit(model)



import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', required=True, type=str, help='Config file for run.')
    parser.add_argument('--gpus', type=int, default=1, required=False, help='Number of gpus.')
    parser.add_argument('-b', '--batch_size', type=int, required=False, help='Batch size per gpu.')
    parser.add_argument('-n', '--run_name', type=str, default='', required=False, help='run name. Overrides run_name in config file.')
    parser.add_argument('-w', '--num_workers', type=int, default=8, required=False, help='Number of workers per device.')

    pargs = parser.parse_args()

    config_file = pargs.config_file
    assert os.path.isfile(config_file), 'Invalid config file:\n%s'%config_file

    args = Hparams()
    args.continue_from_checkpoint = None

    args.load_yaml(config_file)
    if args.continue_from_checkpoint is not None:
        if args.continue_from_checkpoint.endswith('.ckpt'):
            ckpt_path = args.continue_from_checkpoint
        else:
            ckpt_files = get_checkpoints_from_dir(args.continue_from_checkpoint)
            ckpt_path = ckpt_files[-1]
        
        print('loading args from checkpoint...')
        check_args = torch.load(ckpt_path, map_location='cpu')['hyper_parameters']['args']
        check_args.continue_from_checkpoint = ckpt_path
        if hasattr(args, 'overwrite_args'):
            print('\nUpdating args listed in `overwrite_args`. New args:')
            check_args.update(args.overwrite_args)
            print(check_args, end='\n\n', flush=True)
        args = check_args
        epochs = args.epochs

    # run name
    if pargs.run_name != '':
        if hasattr(args, 'run_name'):
            print('Warning: overriding run name in config file with command line argument.')
        args.run_name = pargs.run_name
    if getattr(args, 'run_name', '') in ['', 'None']:
        args.run_name = 'unnamed'

    if platform.system() == 'Windows':
        os.system('export PL_TORCH_DISTRIBUTED_BACKEND=gloo')
    elif not args.run_name.startswith('cluster_'):
        args.run_name = 'cluster_' + args.run_name
        
    args.num_devices = pargs.gpus

    # scale warm period with number of devices (fewer steps per epoch with more devices)
    args.warmup_epochs = int(args.warmup_epochs * args.num_devices)

    trainer_args = Hparams()

    if hasattr(pargs, 'batch_size'):
        if pargs.batch_size is not None:
            args.batch_size = pargs.batch_size # overwrite args.batch size
    assert hasattr(args, 'batch_size'), 'No batch_size given. Pass argument -b.'

    args.accumulate_grad_batches = 1

    args.clip_gradients = None # 'norm', 'val'
    args.gradient_clip_val = 220 if args.clip_gradients is not None else None
    args.gradient_skip_val = -1  # skip update if gradients above this value

    trainer_args.num_workers = pargs.num_workers
    args.trainer_args = trainer_args

    # periodic checkpoints
    args.checkpoint_every_n_epochs = 200
    args.checkpoint_save_top_k = 4
    args.checkpoint_weights_only = True
    args.epochs_early_stop = 150  # stop training early at specified epoch

    main(args)
