import pytorch_lightning as pl

from torch import Tensor


class BinaryClassifier(pl.LightningModule):
    def __init__(self):
        super.__init__()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def training_step(self, batch, batch_idx) -> Tensor:
        raise NotImplementedError()
    
    def test_step(self, batch, batch_idx) -> Tensor:
        raise NotImplementedError()

    def configure_optimizers(self):
        raise NotImplementedError()
