import torch
import pytorch_lightning as pl
from abc import ABC, abstractmethod

class YOLOBaseModel(pl.LightningModule, ABC):
    def __init__(self, config):
        super(YOLOBaseModel, self).__init__()
        self.config = config
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer_name = self.config['training'].get('optimizer', 'adam')
        lr = self.config['training'].get('lr', 1e-3)

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        return optimizer

