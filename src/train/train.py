import os
import sys
if './' not in sys.path:
	sys.path.append('./')
	
from omegaconf import OmegaConf
import argparse

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from ldm.util import instantiate_from_config
from models.util import load_state_dict
from models.logger import ImageLogger
from models.hack import CheckpointCopyCallback
from models.hack import disable_verbosity, count_params, print_all_children

disable_verbosity()

parser = argparse.ArgumentParser(description='ComposeAndConquer Training')
parser.add_argument('--config-path', type=str, default='./configs/cnc_v1.yaml')
parser.add_argument('--learning-rate', type=float, default=1e-5)
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--training-epochs', type=int, default=30)
parser.add_argument('--resume-path', type=str, default='./cnc_v1.ckpt')
parser.add_argument('--logdir', type=str, default='./log_cnc_v1_v100-8/')
parser.add_argument('--log-freq', type=int, default=20)
parser.add_argument('--sd-locked', type=bool, default=True)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--gpus', type=int, default=-1)
parser.add_argument('--persistent-workers', type=bool, default=False)
parser.add_argument('--save-intermediate', type=bool, default=False)
args = parser.parse_args()

def main():
    config_path = args.config_path
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    training_epochs = args.training_epochs
    resume_path = args.resume_path
    default_logdir = args.logdir
    logger_freq = args.log_freq
    sd_locked = args.sd_locked
    num_workers = args.num_workers
    gpus = args.gpus
    persistent_workers = args.persistent_workers
    save_intermediate = args.save_intermediate

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config['model'])

    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked

    dataset = instantiate_from_config(config['data'])
    dataloader = DataLoader(dataset,
                            num_workers=num_workers,
                            batch_size=batch_size,
                            pin_memory=True, shuffle=True, persistent_workers=persistent_workers)

    logger = ImageLogger(batch_frequency=logger_freq, num_local_conditions=2)

    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=logger_freq,
        filename='{epoch}-{step}',
    )
    
    wandb_logger = WandbLogger(project='CnC_v2', job_type='train', save_dir=default_logdir, log_model=True,)

    if save_intermediate:
        print("---------------------------------------")
        print("run will save intermediate checkpoints.")
        print("---------------------------------------")
        checkpoint_copy_callback = CheckpointCopyCallback(default_logdir)
        trainer = pl.Trainer(
            gpus=gpus,
            logger=wandb_logger,
            callbacks=[logger, checkpoint_callback, checkpoint_copy_callback],
            default_root_dir=default_logdir,
            max_epochs=training_epochs,
        )

    else:
        print("-------------------------------------------")
        print("run will NOT save intermediate checkpoints.")
        print("-------------------------------------------")
        trainer = pl.Trainer(
            gpus=gpus,
            logger = wandb_logger,
            callbacks=[logger, checkpoint_callback],
            default_root_dir=default_logdir,
            max_epochs=training_epochs,
        )
    trainer.fit(model,dataloader)

if __name__ == '__main__':
    main()