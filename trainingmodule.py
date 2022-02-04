import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch import Tensor
from models.logistric_regression import LogisticRegression
from fairness import SPD, EOD
from torchmetrics import Accuracy
from collections import OrderedDict


class BinaryClassifier(pl.LightningModule):
    def __init__(self,
                 model: str,
                 input_size: tuple,
                 lr: float=1e-3,
                 weight_decay: float=0.09) -> None:
        """
        A LightningModule for binary classification.

        Args:
            model: the type of model to use
            input_size: the size of the input for model
            lr: the learning rate for training
            weight_decay: the weight decay for training
        """
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
        """
        The forward pass of the model.

        Args:
            x: the input

        Returns: the output

        """
        return self.linear(x)

    def training_step(self, batch, batch_idx) -> OrderedDict:
        """
        The training step for each batch iteration updating each metric.

        Args:
            batch: the current batch
            batch_idx: the index of the batch

        Returns: the loss of the current batch

        """
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

    def test_step(self, batch, batch_idx) -> dict:
        """
        The test step for each batch iteration logging the logits, targets and mask.

        Args:
            batch: the current batch
            batch_idx: the index of the batch

        Returns: a dictionary with the logits, the targets and the mask

        """
        x, y, adv_mask = batch
        logits = self(x)

        return {'logits': logits, 'y': y, 'adv_mask': adv_mask}

    def test_epoch_end(self, outputs) -> dict:
        """
        The process followed at the end of the test epoch calculating every metric.

        Args:
            outputs: the concatenated dictionaries with logits, the targets and the masks

        Returns: a dictionary with the metrics loss, test error, EOD and SPD.

        """
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
        """
        Configuring the optimizer to be used for training.

        Returns:the optimizer

        """
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
        """
        Get the model's parameters.

        Returns: the model's parameters as a tuple
        """
        return self.linear.get_params()

    def get_grads(self):
        """
        Get the model's gradients.

        Returns: the model's gradients as a concatenated tensor
        """
        return self.linear.get_grads()

    def set_params(self, params):
        """
        Set the model's parameters.

        Args:
            params: the parameters to set
        """
        return self.linear.set_params(params=params)

    @property
    def device(self):
        """

        Returns: the device that the model is currently in.

        """
        return self.linear.device
