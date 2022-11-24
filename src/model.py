import numpy as np
from torch import nn
from torchvision import models
from sklearn.metrics import accuracy_score

import timm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score

from .base_model import BaseModel
from .loss import ArcFaceLoss

class ArcMarginProduct(nn.Module):
    """ArcMarginProduct operation.
    Calculates the cosine of the angle between the embeddings and their
    corresponding centers (represented by the weight matrix).
    
    Attributes:
        weight: initialized weight matrix to map embeddings to output classes
        k: Number of subcenter for each class
        out_features: number of output classes
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitializes the weight matrix using xavier_uniform_"""
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        """Perform cosine calculation
        
        Args:
            features: embedding vectors of current minibatch
        Returns:
            A vector of cosine between embeddings and their corresponding centers
            (represented by the weight matrix)
        """
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine

class LivenessModel(BaseModel):
    def __init__(self, backbone_name, backbone_pretrained, embedding_size=512,
                 n_classes=1, n_individuals=1, device='cpu'):
        super(BaseModel, self).__init__()
        
        self.backbone = getattr(models, backbone_name)(weights=backbone_pretrained)
        pool_size = self.backbone.fc.in_features
        self.backbone.classifier = nn.Identity()

        self.neck = nn.Sequential(
                nn.Linear(pool_size, embedding_size, bias=True),
                nn.LayerNorm(embedding_size),
                torch.nn.PReLU()
            )

        self.metric_learning_head = ArcMarginProduct(embedding_size, n_individuals)
        self.clf_head = nn.Linear(embedding_size, n_classes)

        self.device = device

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()
        self.metric_learning_criterion = ArcFaceLoss()

    def forward(self, X):
        batch_size = X.shape[0]
        pooled_features = self.backbone(X)[:,:,0,0]
        embedding = self.neck(pooled_features)
        metric_learning_logits = self.metric_learning_head(embedding)
        y_pred = self.clf_head(embedding)
        return y_pred, metric_learning_logits

    def step(self, X, y, indv_ids):
        X = X.to(self.device)
        y = y.to(self.device)
        indv_ids = indv_ids.to(self.device)
        y_pred, metric_learning_logits = self(X)
        y_pred = y_pred.view(-1)
        clf_loss = self.criterion(y_pred, y)
        metric_learning_loss = self.metric_learning_criterion(metric_learning_logits, indv_ids)
        return clf_loss, metric_learning_loss, y_pred

    def training_step(self, train_batch):
        X, y, indv_ids = train_batch
        clf_loss, metric_learning_loss, y_pred = self.step(X, y, indv_ids)
        loss = clf_loss + metric_learning_loss
        y_prob = y_pred.sigmoid()
        return {'loss': loss, 'clf_loss': clf_loss, 'metric_learning_loss': metric_learning_loss, 'preds':y_prob, 'labels':y}

    def validation_step(self, val_batch):
        X, y, indv_ids = val_batch
        clf_loss, metric_learning_loss, y_pred = self.step(X, y, indv_ids)
        loss = clf_loss + metric_learning_loss
        y_prob = y_pred.sigmoid()
        return {'loss': loss, 'clf_loss': clf_loss, 'metric_learning_loss': metric_learning_loss, 'preds':y_prob, 'labels':y}


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