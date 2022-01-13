import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix
import torch
import torch.nn as nn


class Classifier(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model
        self.accuracy = Accuracy()
        self.conf_matrix = ConfusionMatrix(num_classes=self.model.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.conf_matrix.update(preds, y)

        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_epoch_end(self, outputs):
        print(self.conf_matrix.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
