from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import os
import rnn_utils as utils
import torchmetrics
from torchtext.vocab import GloVe


class Glove_WSD(pl.LightningModule):
    def __init__(self,embedding_dim: int, hidden_dim: int, num_labels: int, layers_num: int,lin_lr:float, lstm_lr: float, embed_dropout: float,lin_dropout: float,lin_wd: float,lstm_wd: float):
        """Init class for WSD classifier with GloVe

        Args:
        embedding_dim (int): LSTM embedding dimension
        hidden_dim (int): LSTM hidden dimension
        num_labels (int): Number of labels
        layers_num (int): LSTM number of layers
        lin_lr        (float): learning rate for the linear layer
        lstm_lr       (float): learning rate for the LSTM
        embed_dropout (float): dropout on the embedding (input of the LSTM)
        lin_dropout   (float): dropout for the linear layer
        lin_wd        (float): weight decay for the linear layer
        lstm_wd       (float): weight decay for the LSTM
        """
        super().__init__()

        
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.lin_lr = lin_lr
        self.embedding_dim = embedding_dim
        self.lstm_lr = lstm_lr
        self.dropout = nn.Dropout(embed_dropout)
        self.lin_dropoout = nn.Dropout(lin_dropout)
        self.lin_wd = lin_wd
        self.lstm_wd = lstm_wd
               
        
        self.word_embeddings = GloVe(name="840B", cache = os.path.join(utils.DIRECTORY_NAME,"glove.840B"),dim = 300)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers_num,
                           bidirectional=utils.BIDIRECTIONAL, batch_first=True,dropout = utils.DROPOUT_LSTM)
        
        
        if (utils.BIDIRECTIONAL):

            self.hidden2labels = nn.Linear(2*hidden_dim, num_labels)
        else:
            self.hidden2labels = nn.Linear(hidden_dim, num_labels)
        
        self.val_metric = torchmetrics.F1Score(
            task="multiclass", num_classes=num_labels, average='micro')
        self.test_metric = torchmetrics.F1Score(
            task="multiclass", num_classes=num_labels, average='micro')

        self.save_hyperparameters()

    def forward(
        self,
        idx:torch.Tensor,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Model forward pass

        Args:
            idx (Tensor): Target word indices
            input_ids (Tensor):Glove input ids

        Returns:
            Tensor: Model predictions
        """
        
        embeds = utils.glove_embedding_tensorization(input_ids,self.word_embeddings)
        embeds = self.dropout(embeds)

        lstm_out, _ = self.lstm(embeds)   
        
        target_vector = utils.get_target_vector(lstm_out, idx)
        target_vector = self.lin_dropoout(target_vector)

        logits = self.hidden2labels(target_vector)

        return logits

    def configure_optimizers(self):
        groups = [
            {
                "params": self.hidden2labels.parameters(),
                "lr": self.lin_lr,
                 "weight_decay": self.lin_wd,
            },
            {
                "params": self.lstm.parameters(),
                "lr": self.lstm_lr,
                 "weight_decay": self.lstm_wd,
            }
        
            
        ]
        optimizer = torch.optim.AdamW(groups)
        return optimizer

    def training_step(self, train_batch, idx) -> STEP_OUTPUT:
        outputs = self(**train_batch)

        loss = F.cross_entropy(outputs.view(-1, self.num_labels),
                               train_batch["labels"].view(-1))
        self.log_dict({'train_loss': loss}, batch_size= utils.GLOVE_BATCH_SIZE,
                      on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, val_batch, idx):
        outputs = self(**val_batch)
        y_pred = outputs.argmax(dim=1)
        
        loss = F.cross_entropy(outputs.view(-1, self.num_labels),
                               val_batch["labels"].view(-1), ignore_index=-100)
        
        self.val_metric(y_pred, val_batch["labels"])
        self.log_dict({'val_loss': loss, 'valid_f1': self.val_metric},
                      batch_size=utils.GLOVE_BATCH_SIZE, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, test_batch, idx):

        outputs = self(**test_batch)
        y_pred = outputs.argmax(dim=1)
        
        loss = F.cross_entropy(outputs.view(-1, self.num_labels),
                               test_batch["labels"].view(-1), ignore_index=-100)

        self.test_metric(y_pred, test_batch["labels"])
        self.log_dict({'test_loss': loss, 'loss_f1': self.test_metric},
                      batch_size=utils.GLOVE_BATCH_SIZE, on_epoch=True, on_step=False, prog_bar=True)
