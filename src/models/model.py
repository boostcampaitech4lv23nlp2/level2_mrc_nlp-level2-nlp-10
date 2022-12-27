import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer


class Model(pl.LightningModule):
    def __init__(self, conf, device, metric_object=None, is_scheduler=False):
        """
        Args:
            conf (config): configuration file
            device (str): gpu device name
            metric_object (function): evaluation metric class
            is_scheduler (bool): scheduler 사용 여부. Defaults to None.
        """
        super().__init__()
        self.save_hyperparameters()

        self._device = device
        self.learning_rate = conf.learning_rate
        self.max_length = conf.max_seq_length
        model_config = AutoConfig.from_pretrained(conf.model_name)
        model_config.num_labels = 2
        self.plm = AutoModelForQuestionAnswering.from_pretrained(
            conf.model_name,
            config=model_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(conf.model_name, max_length=self.max_length)
        self.plm.resize_token_embeddings(self.tokenizer.vocab_size)
        self.metric_object = metric_object
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.is_scheduler = is_scheduler

    def forward(self, batch):
        x = self.plm(**batch)
        return x

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        if self.metric_object:
            logits = (start_logits, end_logits)
            metric = self.metric_object.eval(batch, logits)
            self.log("val_metric", metric)
        return start_logits, end_logits

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        if self.metric_object:
            logits = (start_logits, end_logits)
            metric = self.metric_object.eval(batch, logits)
            self.log("val_metric", metric)
        return logits

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        if self.metric_object:
            predictions = self.metric_object.predict(batch, logits)
        return predictions

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        if self.is_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=1, T_mult=2, eta_min=self.learning_rate * 0.01
            )
            return [optimizer], [lr_scheduler]
        else:
            return optimizer
