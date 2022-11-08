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
        if 'swin' in self.cfg.backbone:
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
        