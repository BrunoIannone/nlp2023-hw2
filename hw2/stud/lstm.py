from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import time
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import os
import lstm_utils as utils
import torchmetrics
import utilz
from allennlp.modules import elmo,elmo_lstm


class Lstm_WSD(pl.LightningModule):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, num_labels: int, layers_num: int, embedding,label_list, lin_lr, elmo_lr, dropout,lin_wd,elmo_wd):

        super().__init__()

        """Init class for the BIO classifier

        Args:
        embedding_dim (int): Embedding dimension
        hidden_dim (int): Hidden dimension
        vocab_size (int): Vocabulary size
        num_labels (int): Number of classes
        layers_num (int): Number of layers of the LSTM
        device (str): Device for calculation
        """
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.label_list = label_list
        self.lin_lr = lin_lr
        self.elmo_lr = elmo_lr
        self.dropout = dropout
        self.lin_wd = lin_wd
        self.elmo_wd = elmo_wd
        #if (embedding):  # note that the vocabulary must have an entry for padding and unk
        #    self.word_embeddings = nn.Embedding.from_pretrained(
        #        embedding, freeze=True)
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers_num,
                           bidirectional=utils.BIDIRECTIONAL, batch_first=True,dropout = utils.DROPOUT_LSTM)
        #self.lstm = elmo_lstm.ElmoLstm(utils.EMBEDDING_DIM,utils.HIDDEN_DIM,2048,utils.LAYERS_NUM,requires_grad=True)

        self.elmo = elmo.Elmo(os.path.join(utilz.DIRECTORY_NAME, "../../model/elmo_2x4096_512_2048cnn_2xhighway_options.json"), os.path.join(
            utilz.DIRECTORY_NAME, "../../model/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"), num_output_representations=2,requires_grad=True,keep_sentence_boundaries=False,do_layer_norm=True)
        if (self.dropout > 0):
            self.dropout_layer = nn.Dropout(self.dropout)

        #if (utils.DROPOUT_EMBED > 0):
        #    self.dropout_embed = nn.Dropout(utils.DROPOUT_EMBED)
        #if (self.dropout > 0):
        #    self.dropout_embed = nn.Dropout(utils.DROPOUT_EMBED)
        
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
        sentence (tuple): (Tensor[padded_sentences], List[lenghts (int)]) N.B.: lenghts refers to the original non padded sentences

        Returns:
            Tensor: Model predictions
        """
        embeds = self.elmo(input_ids)
        #print("Lunghezza embedding:  " + str(embeds['elmo_representations'][-1].size()))
        #time.sleep(10)
        #output_padded, _ = self.lstm(embeds['elmo_representations'][-1])#,embeds['mask'])
        #if(utils.DROPOUT_EMBED > 0):
           # embeds = self.dropout_embed(embeds)
        output_padded = utils.get_senses_vector(embeds['elmo_representations'][-1], idx, None)
        
        # output_padded, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
        #    lstm_out, batch_first=True)
        #embeds = self.word_embeddings(input_ids)
        #if(utils.DROPOUT_EMBED > 0):
         #   embeds = self.dropout_embed(embeds)

        #embeds = torch.nn.utils.rnn.pack_padded_sequence(
           # embeds, sentence[1], batch_first=True, enforce_sorted=False)

        #lstm_out, _ = self.lstm(embeds)

        #output_padded, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
           # lstm_out, batch_first=True)
        if (self.dropout > 0):
            output_padded = self.dropout_layer(output_padded)

        labels_space = self.hidden2labels(output_padded)
        return labels_space

    def configure_optimizers(self):
        groups = [
            {
                "params": self.hidden2labels.parameters(),
                "lr": self.lin_lr,
                 "weight_decay": self.lin_wd,
            },
        
            {
                "params": self.elmo.parameters(),
                "lr": self.elmo_lr,
                "weight_decay": self.elmo_wd,
            }
        ]
        optimizer = torch.optim.AdamW(groups)
        return optimizer

    def training_step(self, train_batch, idx) -> STEP_OUTPUT:
        outputs = self(**train_batch)

        loss = F.cross_entropy(outputs.view(-1, self.num_labels),
                               train_batch["labels"].view(-1))#, ignore_index=-100)
        self.log_dict({'train_loss': loss}, batch_size=utilz.BATCH_SIZE,
                      on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, val_batch, idx):
        outputs = self(**val_batch)
        # print(outputs.size())
        y_pred = outputs.argmax(dim=1)
        # print(val_batch["labels"].size())

        loss = F.cross_entropy(outputs.view(-1, self.num_labels),
                               val_batch["labels"].view(-1), ignore_index=-100)
        self.val_metric(y_pred, val_batch["labels"])
        self.log_dict({'val_loss': loss, 'valid_f1': self.val_metric},
                      batch_size=utilz.BATCH_SIZE, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, test_batch, idx):

        outputs = self(**test_batch)
        y_pred = outputs.argmax(dim=1)
        #predicted_labels = utilz.idx_to_label(
        #    self.label_list, y_pred.tolist())
        # print("RES: " + str(predicted_labels))
        # print(utilz.idx_to_label(
        #            self.label_list, test_batch["labels"].tolist()))
        loss = F.cross_entropy(outputs.view(-1, self.num_labels),
                               test_batch["labels"].view(-1), ignore_index=-100)

        self.test_metric(y_pred, test_batch["labels"])
        self.log_dict({'test_loss': loss, 'loss_f1': self.test_metric},
                      batch_size=utilz.BATCH_SIZE, on_epoch=True, on_step=False, prog_bar=True)
