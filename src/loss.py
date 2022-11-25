import math
import numpy as np
import torch
from torch import nn 
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    """ArcFace loss module as in: https://arxiv.org/abs/1801.07698
    Attributes:
        weight: 
            class weight mapping
            used to put more emphasis on some classes
        reduction:
            how an array of losses for each pair of (target, prediction)
            will be reduced to a single loss. 
        class_weight_norm: 
            only has effect when weight is not None, it takes either 'batch' or 'global' 
            indicates how the minibatch weighted loss should be reduced
        crit: the final torch loss module for classification
        s: ArcFace scale parameter
        m: ArcFace margin angle parameter in radian
        cos_m: cosine value of the margin angle
        sin_m: sine value of the margin angle
        th: cosine value of pi-m
        mm: sine value of (pi-m) multiplied by m
    """
    def __init__(self, s=45.0, m=0.1, weight=None, 
                reduction="mean",class_weights_norm=None,
                device='cuda:0'):
        """
        Args:
            weight: 
                class weight mapping
                used to put more emphasis on some classes
            reduction:
                how an array of losses for each pair of (target, prediction)
                will be reduced to a single loss. 
            class_weight_norm: 
                only has effect when weight is not None, it takes either 'batch' or 'global' 
                indicates how the minibatch weighted loss should be reduced
            device: a device to perform computations on
        Raises:
            ValueError: An error occur when 
        """
        super().__init__()

        self.weight = weight
        self.reduction = reduction
        self.class_weights_norm = class_weights_norm
        
        self.crit = nn.CrossEntropyLoss(reduction="none")   
        
        if s is None:
            self.s = torch.nn.Parameter(torch.tensor([45.], requires_grad=True, device=device))
        else:
            self.s = s

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, logits, labels):
        """Returns the loss between logits and labels
        Args:
            logits: raw output of the model
            labels: ground truth class indices
        """
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        labels2 = torch.zeros_like(cosine)
        labels2.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (labels2 * phi) + ((1.0 - labels2) * cosine)

        s = self.s

        output = output * s
        loss = self.crit(output, labels)

        if self.weight is not None:
            w = self.weight[labels].to(logits.device)

            loss = loss * w
            if self.class_weights_norm == "batch":
                loss = loss.sum() / w.sum()
            if self.class_weights_norm == "global":
                loss = loss.mean()
            else:
                raise ValueError("When weight is not None. class_weights_norm takes either 'batch' or 'global' ")
            return loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError('No such value for reduction')
        return loss    