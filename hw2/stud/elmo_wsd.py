from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import os
import lstm_utils as utils
import torchmetrics
import utilz
from allennlp.modules import elmo


class Elmo_WSD(pl.LightningModule):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, num_labels: int, layers_num: int, embedding,label_list, lin_lr, elmo_lr, embed_dropout,lin_dropout,lin_wd,elmo_wd):
        """Init class for the WSD classifier with elmo

        Args:

        hidden_dim    (int): Hidden dimension
        num_labels    (int): Number of classes
        embedding_dim (int): useless parameter, left for error
        vocab_size    (int): useless parameter, left for error
        layers_num    (int): useless parameter, left for error
        embedding     (int): useless parameter, left for error 
        label_list    (int): useless parameter, left for error
        lin_dropout   (float): dropout for the linear layer
        lin_lr        (float): learning rate for the linear layer
        elmo_lr       (float): learning rate for ELMo
        lin_wd        (float): weight decay for the linear layer
        elmo_wd       (float): weight decay for ELMo
        """
        super().__init__()

        
        
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.lin_lr = lin_lr
        self.elmo_lr = elmo_lr
        self.lin_dropout = nn.Dropout(lin_dropout)
        self.lin_wd = lin_wd
        self.elmo_wd = elmo_wd
        
        self.elmo = elmo.Elmo(os.path.join(utilz.DIRECTORY_NAME, "../../model/elmo_2x4096_512_2048cnn_2xhighway_options.json"), os.path.join(
            utilz.DIRECTORY_NAME, "../../model/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"), num_output_representations=2,requires_grad=True,keep_sentence_boundaries=False,do_layer_norm=True)
        
        self.hidden2labels = nn.Linear(hidden_dim, num_labels)
        
        self.val_metric = torchmetrics.F1Score(
            task="multiclass", num_classes=num_labels, average='micro')
        self.test_metric = torchmetrics.F1Score(
            task="multiclass", num_classes=num_labels, average='micro')

        self.save_hyperparameters()

    def forward(
        self,
        idx,
        input_ids: torch.Tensor = None,
        
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
        compute_predictions: bool = False,
        compute_loss: bool = True,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Model forward pass

        Args:
        input_ids (Tensor): tensor containing input tensor of word ids
        idx (Tensor): tensor containing indices for each sentence targets
        Returns:
            Tensor: Model predictions
        """
        embeds = self.elmo(input_ids)
        
        output_padded = utils.get_senses_vector(embeds['elmo_representations'][-1], idx, None)
        output_padded = self.lin_dropout(output_padded)

        labels_space = self.hidden2labels(output_padded)
        
        return labels_space

    def configure_optimizers(self):
        groups = [
            {
                "params": self.hidden2labels.parameters(),
                "lr": self.lin_lr,
                 "weight_decay": self.lin_wd
            },
            {
                "params": self.elmo.parameters(),
                "lr": self.elmo_lr,
                 "weight_decay": self.elmo_wd
            }
        
            
        ]
        optimizer = torch.optim.AdamW(groups)
        return optimizer

    def training_step(self, train_batch, idx) -> STEP_OUTPUT:
        outputs = self(**train_batch)

        loss = F.cross_entropy(outputs.view(-1, self.num_labels),
                               train_batch["labels"].view(-1))
        self.log_dict({'train_loss': loss}, batch_size=utilz.BATCH_SIZE,
                      on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, val_batch, idx):
        outputs = self(**val_batch)
        y_pred = outputs.argmax(dim=1)
        
        loss = F.cross_entropy(outputs.view(-1, self.num_labels),
                               val_batch["labels"].view(-1), ignore_index=-100)
        
        self.val_metric(y_pred, val_batch["labels"])
        self.log_dict({'val_loss': loss, 'valid_f1': self.val_metric},
                      batch_size=utilz.BATCH_SIZE, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, test_batch, idx):

        outputs = self(**test_batch)
        y_pred = outputs.argmax(dim=1)
        
        loss = F.cross_entropy(outputs.view(-1, self.num_labels),
                               test_batch["labels"].view(-1), ignore_index=-100)

        self.test_metric(y_pred, test_batch["labels"])
        self.log_dict({'test_loss': loss, 'loss_f1': self.test_metric},
                      batch_size=utilz.BATCH_SIZE, on_epoch=True, on_step=False, prog_bar=True)
