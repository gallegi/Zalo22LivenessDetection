from copy import deepcopy
from abc import abstractmethod, ABC
import torch
from torch import nn

class BaseModel(nn.Module, ABC):
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

class ModelEmaV2(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        model.cpu()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        model.to(device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        model.cpu()
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)
        model.to(self.device) # back to device

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)