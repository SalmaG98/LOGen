import os
import click
from os.path import join, dirname, abspath
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np
import torch
import yaml
from logen.datasets.dataset_mapper import dataloaders
from logen.models.diffuser import Diffuser
from logen.modules.callbacks import GenerationEvalCallback

def set_deterministic():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

def configure_cuda(ngpus=1):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # ngpus-1 to leave out one gpus fully reserved for generation eval callback
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(ngpus-1)))

@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/config.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
@click.option('--resdir',
              '-r',
              type=str,
              help='path to save generation results in eval callbacks.',
              default=None)
@click.option('--test', '-t', is_flag=True, help='test mode')

def main(config, weights, checkpoint, resdir, test):
    if not test:
        set_deterministic()

    cfg = yaml.safe_load(open(config))

    configure_cuda(cfg['train']['n_gpus'])
    cfg['train']['n_gpus']-=1
    
    #Load data and model
    if weights is None:
        model = Diffuser(cfg)
    else:
        if test:
            ckpt_cfg = yaml.safe_load(open(config))
            cfg = ckpt_cfg

        model = Diffuser(cfg)
        model = model.load_from_checkpoint(weights, hparams=cfg)

    dl = cfg['data']['dataloader']
    data = dataloaders[dl](cfg)

    #Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    periodic_checkpoint_saver = ModelCheckpoint(
                                dirpath=f'{resdir}/checkpoints/'+cfg['experiment']['id'],
                                filename='{epoch:d}',
                                save_last=True, 
                                every_n_epochs=25, 
                                enable_version_counter=False, #new run checkpoint epoch=$epoch-last.ckpt overwites the previous one
                                save_top_k=-1        # creates last.ckpt
                            )
    
    best_checkpoint_saver = ModelCheckpoint(
                                dirpath=f'{resdir}/checkpoints/'+cfg['experiment']['id'],
                                filename='best',
                                monitor='val/jsd_mean',
                                save_top_k=1,          # keep only best
                                mode='min',            # or 'max' depending on metric
                                # save_last=True        # creates last.ckpt
                            )

    periodic_checkpoint_saver.CHECKPOINT_NAME_LAST = "{epoch}-last"
    
    model_type = 'map' if cfg['model']['map_shape_to_one'] else 'no_map'

    gen_eval_cb = GenerationEvalCallback(
        project_root='./',
        gen_args=[
            model_type
        ],
        eval_scripts_dir='scripts/',
        eval_args={
            "evaluate_cd_emd_logen.sh": [model_type, '3'],
            "evaluate_fid_gens.sh": [model_type, '3'],
            "evaluate_jsd_gens.sh": [model_type, '3'],
            "evaluate_kid_gens.sh": [model_type, '3'],
            "evaluate_nn_cov_logen.sh": [model_type, '3'],
            "evaluate_pointnet_acc_gens.sh": [model_type, '3'],
        },
        run_every_epochs=50,
    )

    tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg['experiment']['id'],
                                             default_hp_metric=False)
    #Setup trainer
    if torch.cuda.device_count() > 1:
        # cfg['train']['n_gpus'] = torch.cuda.device_count()
        trainer = Trainer(
                        devices=cfg['train']['n_gpus'],
                        logger=tb_logger,
                        log_every_n_steps=100,
                        max_epochs= cfg['train']['max_epoch'],
                        callbacks=[lr_monitor, 
                                   periodic_checkpoint_saver, 
                                   best_checkpoint_saver,
                                   gen_eval_cb],
                        check_val_every_n_epoch=5,
                        num_sanity_val_steps=2,
                        limit_val_batches=2,
                        accelerator='gpu',
                        strategy="ddp_find_unused_parameters_true"
                        )
    else:
        trainer = Trainer(
                        accelerator='gpu',
                        devices=cfg['train']['n_gpus'],
                        logger=tb_logger,
                        log_every_n_steps=100,
                        max_epochs= cfg['train']['max_epoch'],
                        callbacks=[lr_monitor, 
                                   periodic_checkpoint_saver, 
                                   best_checkpoint_saver,
                                   gen_eval_cb],                        
                        check_val_every_n_epoch=10,
                        num_sanity_val_steps=2,
                        limit_val_batches=2,
                )


    # Train!
    if test:
        print('TESTING MODE')
        trainer.test(model, data)
    else:
        print('TRAINING MODE')
        trainer.fit(model, data, ckpt_path=checkpoint)

if __name__ == "__main__":
    main()
