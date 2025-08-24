from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Any

from torch import nn
from torch.utils.data import DataLoader


@dataclass
class BaseConfig(ABC):
    config: object = None
    train_dl: DataLoader = None
    test_list: object = None
    model: nn.Module = None
    train_one_epoch: Callable[..., Any] = None
    val_one_epoch: Callable[..., Any] = None

    def __post_init__(self):
        config = self.set_config()
        args = self.parse_args()
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

        self.config = args
        self.train_dl = self.set_train_dl()
        self.val_list = self.set_val_list()
        self.model = self.set_model()
        self.train_one_epoch = self.set_train_one_epoch()
        self.val_one_epoch = self.set_val_one_epoch()

    @abstractmethod
    def set_config(self):
        pass

    @abstractmethod
    def parse_args(self):
        pass

    @abstractmethod
    def set_train_dl(self):
        pass

    @abstractmethod
    def set_val_list(self):
        pass

    @abstractmethod
    def set_model(self):
        pass

    @abstractmethod
    def set_val_one_epoch(self):
        pass

    @abstractmethod
    def set_train_one_epoch(self):
        pass

    def get_config(self):
        return self.config

    def get_train_dl(self):
        return self.train_dl

    def get_test_list(self):
        return self.test_list

    def get_model(self):
        return self.model

    def get_train_one_epoch(self):
        return self.train_one_epoch

    def get_val_one_epoch(self):
        return self.val_one_epoch

    def get_all(self, return_dict=False):
        if return_dict:
            return {
                "config": self.config,
                "train_dl": self.train_dl,
                "val_list": self.val_list,
                "model": self.model,
                "train_one_epoch": self.train_one_epoch,
                "val_one_epoch": self.val_one_epoch
            }
        return self.config, self.train_dl, self.val_list, self.model, self.train_one_epoch, self.val_one_epoch
