import os
import argparse
import importlib
import pandas as pd
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from src.general import seed_torch
from src.dataset import LivenessDataset
from src.model import LivenessLit

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--config', type=str, default='config_v1',
                    help='config file to run an experiment')
parser.add_argument('--model_dir', type=str, default='./models',
                    help='config file to run an experiment')

args = parser.parse_args()

config_module = importlib.import_module(f'configs.{args.config}')
CFG = config_module.CFG

CFG.model_dir = args.model_dir

seed_torch(CFG.seed) # set initial seed

CFG.output_dir_name = CFG.version_note + '_' + CFG.backbone.replace('/', '_') 
CFG.output_dir = os.path.join(args.model_dir, CFG.output_dir_name)

df = pd.read_csv(CFG.metadata_file)

os.makedirs(CFG.output_dir, exist_ok=True)

for valid_fold in CFG.run_folds:
    print(f'================= Training fold {valid_fold} ================')
    seed_torch(CFG.seed) # set seed each time a fold is run

    train_df = df[df['fold']!=valid_fold].reset_index(drop=True)
    valid_df = df[df['fold']==valid_fold].reset_index(drop=True)

    if(CFG.sample):
        train_df = train_df.sample(CFG.sample).reset_index(drop=True)
        valid_df = valid_df.sample(CFG.sample).reset_index(drop=True)

    CFG.num_train_examples = len(train_df)

    # Defining DataSet
    train_dataset = LivenessDataset(CFG, train_df, CFG.train_video_dir, CFG.train_transforms)
    valid_dataset = LivenessDataset(CFG, valid_df, CFG.train_video_dir, CFG.val_transforms)

    batch_size = CFG.batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True,
                                               pin_memory=True,drop_last=True,num_workers=CFG.num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size,num_workers=CFG.num_workers,
                                               shuffle=False,pin_memory=True,drop_last=False)

    CFG.steps_per_epoch = int(len(train_df) / CFG.batch_size)
    CFG.num_train_steps = int(len(train_df) / CFG.batch_size * CFG.epochs)
    lit_module = LivenessLit(CFG)

    # logger = CSVLogger(CFG.output_dir, name=f'fold{valid_fold}')
    logger = CometLogger(api_key=CFG.comet_api_key, project_name=CFG.comet_project_name, experiment_name=CFG.output_dir_name + f'_fold{valid_fold}')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpointer = ModelCheckpoint(
         dirpath=os.path.join(CFG.output_dir, f'fold{valid_fold}'),
         filename='{epoch}-{val_loss:.3f}-{val_acc:.3f}',
         monitor='val_loss',
         verbose=True,
         save_weights_only=True
    )
    trainer = Trainer(default_root_dir=CFG.output_dir, precision=16, max_epochs=CFG.epochs,
                     check_val_every_n_epoch=1, enable_checkpointing=True,
                     log_every_n_steps=CFG.steps_per_epoch,
                     logger=logger,
                     callbacks=[lr_monitor, checkpointer],
                     accelerator=CFG.accelerator, devices=CFG.devices)
    trainer.fit(lit_module, train_loader, valid_loader)
    # break