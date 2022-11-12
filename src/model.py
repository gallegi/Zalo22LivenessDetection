import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score

import timm
import pytorch_lightning as pl

class LivenessLit(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(self.cfg.backbone, pretrained=True)
        # self.backbone.load_state_dict(torch.load(backbone_pretrained))
        
        if 'nfnet' in self.cfg.backbone:
            clf_in_feature = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Linear(clf_in_feature, 1)
        elif 'swin' in self.cfg.backbone:
            clf_in_feature = self.backbone.head.in_features
            self.backbone.head = nn.Linear(clf_in_feature, 1)
        elif 'resnet' in self.cfg.backbone:
            clf_in_feature = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(clf_in_feature, 1)
        else:
            clf_in_feature = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(clf_in_feature, 1)

        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, X):
        return  self.backbone(X)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        optimizer = AdamW(self.backbone.parameters(), lr=self.cfg.init_lr, eps=self.cfg.eps, betas=self.cfg.betas)
        num_train_steps = int(self.cfg.num_train_examples / self.cfg.batch_size * self.cfg.epochs)

        #Defining LR SCheduler
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_train_steps, eta_min=self.cfg.min_lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                # "monitor": "metric_to_track",
                # "frequency": "indicates how often the metric is updated"
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def step(self, X, y):
        X = X.to(self.device)
        y = y.to(self.device)
        y_pred = self(X).view(-1)
        loss = self.criterion(y_pred, y)
        return loss, y_pred

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        loss, y_pred = self.step(X, y)
        self.log('train_loss', loss)
        return {'loss': loss, 'preds':y_pred, 'labels':y}

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        loss, y_pred = self.step(X, y)
        self.log('val_loss', loss)
        y_prob = y_pred.sigmoid()
        return {'loss': loss, 'preds':y_prob, 'labels':y}

    def predict_step(self, test_batch, batch_idx):
        X = test_batch[0]
        X = X.to(self.device)
        y_pred = self(X).view(-1)
        y_prob = y_pred.sigmoid()
        return y_prob

    def compute_metrics(self, outputs):
        all_preds = np.concatenate([out['preds'].detach().cpu().numpy() for out in outputs])
        all_labels = np.concatenate([out['labels'].detach().cpu().numpy() for out in outputs])
        all_preds = (all_preds > 0.5).astype(int)
        acc = float(accuracy_score(y_true=all_labels, y_pred=all_preds))
        return acc

    def training_epoch_end(self, training_step_outputs):
        train_acc = self.compute_metrics(training_step_outputs)
        self.log('train_acc', train_acc)
        
    def validation_epoch_end(self, validation_step_outputs):
        val_acc = self.compute_metrics(validation_step_outputs)
        self.log('val_acc', val_acc)
        


import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import gc
from sklearn.metrics import roc_auc_score, accuracy_score

import timm
from torch import nn

class LivenessModel(nn.Module):
    def __init__(self, backbone_name, backbone_pretrained, n_classes=1):
        super(LivenessModel, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=backbone_pretrained)
        # self.backbone.load_state_dict(torch.load(backbone_pretrained))
        
        if 'nfnet' in backbone_name:
            clf_in_feature = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Linear(clf_in_feature, n_classes)
        elif 'resnet' in backbone_name:
            clf_in_feature = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(clf_in_feature, n_classes)
        else:
            clf_in_feature = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(clf_in_feature, n_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        
        return x
    
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def train_valid_fn(dataloader,model,criterion,optimizer=None,scaler=None,device='cuda:0',scheduler=None,epoch=0,mode='train'):
    '''Perform model training'''
    if(mode=='train'):
        model.train()
    elif(mode=='valid'):
        model.eval()
    else:
        raise ValueError('No such mode')
        
    loss_score = AverageMeter()
    
    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, batch in tk0:
        if(mode=='train'):
            optimizer.zero_grad()
            
        inps, labels = batch
        inps = inps.to(device)
        labels = labels.to(device)
        outputs = model(inps).view(-1)

        with torch.cuda.amp.autocast():
            loss = criterion(outputs, labels)
        
        if(mode=='train'):
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        loss_score.update(loss.detach().cpu().item(), dataloader.batch_size)
        if(mode=='train'):
            tk0.set_postfix(Loss_Train=loss_score.avg, Epoch=epoch, LR=optimizer.param_groups[0]['lr'])
        elif(mode=='valid'):
            tk0.set_postfix(Loss_Valid=loss_score.avg, Epoch=epoch)
    
        
        del batch, inps, outputs, labels, loss
        torch.cuda.empty_cache()
        
    if(mode=='train'):
        if(scheduler.__class__.__name__ == 'CosineAnnealingWarmRestarts'):
            scheduler.step(epoch=epoch)
        elif(scheduler.__class__.__name__ == 'ReduceLROnPlateau'):
            scheduler.step(loss_score.avg)
        elif(scheduler.__class__.__name__ == 'GradualWarmupScheduler'):
            scheduler.step()
    
    return loss_score.avg

def compute_auc_and_accuracy(dataloader, model, device='cuda:0', conf_thresh=0.5):
    model.eval()
    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    batch_preds=[]
    batch_labels=[]
    for i, batch in tk0:
        inps, lbls = batch
        inps = inps.to(device)
        with torch.no_grad():
            outputs = model(inps).view(-1)
            probs = torch.sigmoid(outputs)
        batch_preds.append(probs.cpu().numpy())
        batch_labels.append(lbls.numpy())
        
        del batch, inps, outputs, probs
        torch.cuda.empty_cache()
        
    predictions = np.concatenate(batch_preds)
    labels = np.concatenate(batch_labels)
    auc = roc_auc_score(y_true=labels, y_score=predictions, multi_class='ovo', average='macro')
    acc = accuracy_score(y_true=labels, y_pred=(predictions >= conf_thresh).astype(int))
    return auc, acc