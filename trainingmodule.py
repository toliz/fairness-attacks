import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch import Tensor
from models.logistric_regression import LogisticRegression
from fairness import SPD, EOD
from torchmetrics import Accuracy
from collections import OrderedDict


class BinaryClassifier(pl.LightningModule):
    """
    Lighting Module for binary classification
    """
    def __init__(self,
                 model: str,
                 input_size: tuple,
                 lr: float=1e-3,
                 weight_decay: float=0.09) -> None:
        super().__init__()
        self.save_hyperparameters()

        if model == 'LogisticRegression':
            assert len(input_size) == 1, "Logistic regression expected 1D input"
            self.linear = LogisticRegression(input_size=input_size[0])
        else:
            raise ValueError("Unknown model name")

        self.spd = SPD()
        self.eod = EOD()
        self.acc = Accuracy()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

    def training_step(self, batch, batch_idx) -> OrderedDict:
        # Forward pass
        x, y, _ = batch
        logits = self(x)
        predicts = self.get_predictions(logits)

        # Metrics
        loss = self.loss(logits, y.float())
        acc = self.acc(predicts, y)

        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Forward pass
        x, y, _ = batch
        logits = self(x)
        predicts = self.get_predictions(logits)

        # Metrics
        loss = self.loss(logits, y.float())
        acc = self.acc(predicts, y)

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx) -> dict:
        x, y, adv_mask = batch
        logits = self(x)

        return {'logits': logits, 'y': y, 'adv_mask': adv_mask}

    def test_epoch_end(self, outputs) -> dict:
        y = torch.cat([subdict['y'] for subdict in outputs])
        logits = torch.cat([subdict['logits'] for subdict in outputs])
        adv_mask = torch.cat([subdict['adv_mask'] for subdict in outputs])
        predicts = self.get_predictions(logits)

        spd = self.spd(predicts, adv_mask)
        eod = self.eod(predicts, y, adv_mask)
        loss = self.loss(logits, y.float())
        acc = self.acc(predicts, y)

        self.log('test_error', 1 - acc, on_step=False, on_epoch=True)
        self.log('EOD', eod, on_step=False, on_epoch=True)
        self.log('SPD', spd, on_step=False, on_epoch=True)

        return {'loss': loss, 'test_error': 1 - acc, 'EOD': eod, 'SPD': spd}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

    @staticmethod
    def get_predictions(logits: Tensor) -> Tensor:
        """
        Compute predictions from logits.
        Prediction = 1 if logit > 0
        Prediction = 0 if logit <= 0
        Args:
            logits: logits from model

        Returns: predictions
        """
        return torch.heaviside(logits, torch.tensor(0).float()).int().reshape(-1)

    def get_params(self):
        return self.linear.get_params()

    def get_grads(self):
        return self.linear.get_grads()

    def set_params(self, params):
        return self.linear.set_params(params=params)

    @property
    def device(self):
        return self.linear.device
