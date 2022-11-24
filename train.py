import os
import argparse
import importlib
import pandas as pd
import torch

from warmup_scheduler import GradualWarmupScheduler
from torch.optim import AdamW

from src.general import seed_everything
from src.dataset import LivenessDataset
from src.model import LivenessModel
from src.trainer import Trainer

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--config', type=str, default='config_v1',
                    help='config file to run an experiment')
parser.add_argument('--model_dir', type=str, default='./models',
                    help='config file to run an experiment')

args = parser.parse_args()

config_module = importlib.import_module(f'configs.{args.config}')
CFG = config_module.CFG

CFG.model_dir = args.model_dir

CFG.output_dir_name = CFG.version_note + '_' + CFG.backbone.replace('/', '_') 
CFG.output_dir = os.path.join(args.model_dir, CFG.output_dir_name)

if not torch.cuda.is_available():
    CFG.device = 'cpu'

df = pd.read_csv(CFG.metadata_file)
df = df[df.set == 'train'].reset_index(drop=True)

n_individuals = df.individual_id.nunique()

os.makedirs(CFG.output_dir, exist_ok=True)

for valid_fold in CFG.run_folds:
    print(f'================= Training fold {valid_fold} ================')
    seed_everything(CFG.seed) # set seed each time a fold is run

    train_df = df[df['fold']!=valid_fold].reset_index(drop=True)
    valid_df = df[df['fold']==valid_fold].reset_index(drop=True)

    if(CFG.sample):
        train_df = train_df.sample(CFG.sample).reset_index(drop=True)
        valid_df = valid_df.sample(CFG.sample).reset_index(drop=True)

    CFG.num_train_examples = len(train_df)

    # Defining DataSet
    train_dataset = LivenessDataset(CFG, train_df, CFG.train_video_dir, CFG.train_transforms)
    val_dataset = LivenessDataset(CFG, valid_df, CFG.train_video_dir, CFG.val_transforms)

    batch_size = CFG.batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True,
                                               pin_memory=True,drop_last=True,num_workers=CFG.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,num_workers=CFG.num_workers,
                                               shuffle=False,pin_memory=True,drop_last=False)

    # Model
    model = LivenessModel(CFG.backbone, backbone_pretrained=CFG.pretrained_weights, embedding_size=CFG.embedding_size,
                          n_individuals=n_individuals, device=CFG.device)
    model.to(CFG.device)
    
    # Optimizer and scheduler
    optim = AdamW(model.parameters(), betas=CFG.betas, lr=CFG.init_lr/CFG.warmup_factor, weight_decay=CFG.weight_decay)

    num_training_steps = CFG.epochs * len(train_loader)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, CFG.epochs-1)
    lr_scheduler = GradualWarmupScheduler(optim, multiplier=CFG.warmup_factor, total_epoch=1, after_scheduler=scheduler_cosine)
    
    trainer = Trainer(CFG, model, train_loader, val_loader,
                    optimizer=optim, lr_scheduler=lr_scheduler, fold=valid_fold)

    trainer.fit()