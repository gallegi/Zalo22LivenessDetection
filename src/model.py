import numpy as np
from torch import nn
from sklearn.metrics import accuracy_score

import timm
import numpy as np
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score

import timm
from torch import nn
from abc import abstractmethod, ABC

class Model(nn.Module, ABC):
    @abstractmethod
    def training_step():
        raise NotImplementedError()

    @abstractmethod
    def validation_step():
        raise NotImplementedError()

    def training_epoch_end(self):
        return dict()

    def validation_epoch_end(self):
        return dict()
        

class LivenessModel(Model):
    def __init__(self, backbone_name, backbone_pretrained, n_classes=1, device='cpu'):
        super(Model, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=backbone_pretrained)
        
        if 'nfnet' in backbone_name:
            clf_in_feature = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Linear(clf_in_feature, n_classes)
        elif 'resnet' in backbone_name:
            clf_in_feature = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(clf_in_feature, n_classes)
        else:
            clf_in_feature = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(clf_in_feature, n_classes)
        
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


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)