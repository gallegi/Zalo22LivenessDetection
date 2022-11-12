import os
import argparse
import importlib
import pandas as pd
import torch
from torch import nn

from warmup_scheduler import GradualWarmupScheduler
from torch.optim import AdamW

from src.general import seed_torch
from src.dataset import LivenessDataset
from src.model import LivenessModel, compute_auc_and_accuracy, train_valid_fn
from src.general import seed_torch, init_progress_dict, log_to_progress_dict, save_progress, log_and_print, get_logger

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

    
    # Model
    model = LivenessModel(CFG.backbone, backbone_pretrained=True)
    model.to(CFG.device)
    # print(model)

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()

    # Optimizer and scheduler
    optim = AdamW(model.parameters(), lr=CFG.init_lr/CFG.warmup_factor)

    num_training_steps = CFG.epochs * len(train_loader)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, CFG.epochs-1)
    lr_scheduler = GradualWarmupScheduler(optim, multiplier=CFG.warmup_factor, total_epoch=1, after_scheduler=scheduler_cosine)
    
    # Logging
    logger = get_logger(
        name = f'training_log_fold{valid_fold}.txt',
        path=os.path.join(CFG.output_dir, f'training_log_fold{valid_fold}.txt')
    )

    best_valid_loss = 9999
    best_valid_ep = 0
    patience = CFG.patience

    progress_dict = init_progress_dict(['loss', 'AUC', 'accuracy'])
    
    for epoch in range(0, CFG.epochs):

        # =============== Training ==============
        train_loss = train_valid_fn(train_loader,model,criterion,optimizer=optim,device=CFG.device,
                                    scheduler=lr_scheduler,scaler=scaler,epoch=epoch,mode='train')
        valid_loss = train_valid_fn(valid_loader,model,criterion,device=CFG.device,epoch=epoch,
                                    scaler=scaler,mode='valid')

        # =============== Evaluation =================
        train_auc, train_acc = compute_auc_and_accuracy(train_loader, model, CFG.device)
        valid_auc, valid_acc = compute_auc_and_accuracy(valid_loader, model, CFG.device)
        current_lr = optim.param_groups[0]['lr']
        log_line = f'Model: {CFG.output_dir_name}. Epoch: {epoch}. '
        log_line += f'Train loss:{train_loss} - Valid loss: {valid_loss}. '
        log_line += f'Train AUC:{train_auc} - Valid AUC: {valid_auc}. '
        log_line += f'Train accuracy:{train_acc} - Valid accuracy: {valid_acc}. '
        log_line += f'Lr: {current_lr}.'

        log_and_print(logger, log_line)

        metric_dict = {'train_loss':train_loss,'valid_loss':valid_loss,'train_AUC':train_auc, 'valid_AUC':valid_auc,
                        'train_accuracy':train_acc, 'valid_accuracy':valid_acc}
        progress_dict = log_to_progress_dict(progress_dict, metric_dict)

        # plot figure and save the progress chart
        save_progress(progress_dict, CFG.output_dir, CFG.output_dir_name, valid_fold, show=False)

        # plot figure and save the progress chart
        save_progress(progress_dict, CFG.output_dir, CFG.output_dir_name, valid_fold, show=False)

        if(valid_loss < best_valid_loss):
            best_valid_loss = valid_loss
            best_valid_ep = epoch
            patience = CFG.patience # reset patience

            # save model
            name = os.path.join(CFG.output_dir, 'Fold%d_%s_ValidLoss%03.03f_ValidAcc%03.03f_Ep%02d.pth'%(valid_fold, CFG.output_dir_name, valid_loss, valid_acc, epoch))
            log_and_print(logger, 'Saving model to: ' + name)
            torch.save(model.state_dict(), name)
        else:
            patience -= 1
            log_and_print(logger, 'Decrease early-stopping patience by 1 due valid loss not decreasing. Patient='+ str(patience))

        if(patience == 0):
            log_and_print(logger, 'Early stopping patience = 0. Early stop')
            break