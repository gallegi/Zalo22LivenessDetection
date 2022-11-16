from abc import abstractmethod, ABC
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