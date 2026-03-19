import torch
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
import pytorch_lightning as pl

class TFTModel(pl.LightningModule):
    def __init__(self, config, training_dataset=None):
        super().__init__()
        self.config = config
        self.model = TemporalFusionTransformer.from_dataset(
            training=training_dataset,
            hidden_size=config['model']['hidden_size'],
            attention_head_size=config['model']['attention_head_size'],
            dropout=config['model']['dropout'],
            hidden_continuous_size=config['model']['hidden_continuous_size'],
            output_size=len(config['model']['quantiles']),
            loss=QuantileLoss(quantiles=config['model']['quantiles']),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.model.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.model.loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['model']['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }