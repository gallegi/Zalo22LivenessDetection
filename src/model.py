import numpy as np
from torch import nn
from torchvision import models
from sklearn.metrics import accuracy_score

import timm
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score

from .base_model import BaseModel

class LivenessModel(BaseModel):
    def __init__(self, backbone_name, backbone_pretrained, n_classes=1, device='cpu'):
        super(BaseModel, self).__init__()
        # self.backbone = timm.create_model(backbone_name, pretrained=backbone_pretrained)
        
        # if 'swin' in backbone_name:
        #     clf_in_feature = self.backbone.head.in_features
        #     self.backbone.head = nn.Linear(clf_in_feature, n_classes)
        # elif 'nfnet' in backbone_name:
        #     clf_in_feature = self.backbone.head.fc.in_features
        #     self.backbone.head.fc = nn.Linear(clf_in_feature, n_classes)
        # elif 'resnet' in backbone_name:
        #     clf_in_feature = self.backbone.fc.in_features
        #     self.backbone.fc = nn.Linear(clf_in_feature, n_classes)
        # else:
        #     clf_in_feature = self.backbone.classifier.in_features
        #     self.backbone.classifier = nn.Linear(clf_in_feature, n_classes)
        
        self.backbone = models.convnext_large(pretrained=backbone_pretrained)
        clf_in_feature = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(clf_in_feature, n_classes)

        self.device = device

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, X):
        batch_size = X.shape[0]
        y = self.backbone(X)
        return y

    def step(self, X, y):
        X = X.to(self.device)
        y = y.to(self.device)
        y_pred = self(X).view(-1)
        loss = self.criterion(y_pred, y)
        return loss, y_pred

    def training_step(self, train_batch):
        X, y = train_batch
        loss, y_pred = self.step(X, y)
        return {'loss': loss, 'preds':y_pred, 'labels':y}

    def validation_step(self, val_batch):
        X, y = val_batch
        loss, y_pred = self.step(X, y)
        y_prob = y_pred.sigmoid()
        return {'loss': loss, 'preds':y_prob, 'labels':y}

    def compute_metrics(self, outputs):
        all_preds = np.concatenate([out['preds'].detach().cpu().numpy() for out in outputs])
        all_labels = np.concatenate([out['labels'].detach().cpu().numpy() for out in outputs])
        all_probs = (all_preds > 0.5).astype(int)
        acc = float(accuracy_score(y_true=all_labels, y_pred=all_probs))
        auc = float(roc_auc_score(y_true=all_labels, y_score=all_preds))
        return acc, auc

    def training_epoch_end(self, training_step_outputs):
        train_acc, train_auc = self.compute_metrics(training_step_outputs)
        return {'acc': train_acc, 'AUC': train_auc}
        
    def validation_epoch_end(self, validation_step_outputs):
        val_acc, val_auc = self.compute_metrics(validation_step_outputs)
        return {'acc': val_acc, 'AUC': val_auc}

class LivenessSequenceModel(nn.Module):
    def __init__(self, pretrained_name = 'resnet50'):
        super(LivenessSequenceModel, self).__init__()
        self.backbone = timm.create_model(pretrained_name, pretrained=None)
        if pretrained_name == 'resnet50':
            self.in_feats = self.backbone.fc.in_features
            self.backbone.fc = torch.nn.Identity()
        if pretrained_name == 'darknet53':
            self.in_feats = self.backbone.head.fc.in_features
            self.backbone.head.fc = torch.nn.Identity()
    
        self.lstm = torch.nn.LSTM(self.in_feats, self.in_feats, 2,
                                  bidirectional = True, dropout = 0.3, batch_first = True)
        self.linear = torch.nn.Linear(self.in_feats * 2, 1)
    def forward(self, x):
        b, f, c, h, w = x.shape
        x = torch.reshape(x, (b * f, c, h, w))
        x = self.backbone(x)
        x = torch.reshape(x, (b, f, self.in_feats))
        output, (h, c) = self.lstm(x)
        x = output[:,-1,:]
        x = self.linear(x)
        return x


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)