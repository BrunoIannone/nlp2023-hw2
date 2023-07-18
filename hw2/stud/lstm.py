from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import time
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import lstm_utils as utils
import torchmetrics
import utilz

class Lstm_WSD(pl.LightningModule):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, num_labels: int, layers_num: int, embedding):

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

        if(embedding):  # note that the vocabulary must have an entry for padding and unk
            self.word_embeddings = nn.Embedding.from_pretrained(
                embedding, freeze=True)
        else:
            self.word_embeddings = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers_num,
                        bidirectional=utils.BIDIRECTIONAL, batch_first=True, dropout=utils.DROPOUT_LSTM)

        if(utils.DROPOUT_LAYER > 0):
            self.dropout_layer = nn.Dropout(utils.DROPOUT_LAYER)

        if(utils.DROPOUT_EMBED > 0):
            self.dropout_embed = nn.Dropout(utils.DROPOUT_EMBED)

        if(utils.BIDIRECTIONAL):

            self.hidden2labels = nn.Linear(2*hidden_dim, num_labels)
        else:
            self.hidden2labels = nn.Linear(hidden_dim, num_labels)
        self.val_metric  = torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='micro')
        self.test_metric = torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='micro')

        self.save_hyperparameters()

    def forward(
        self,
        idx,
        word_ids,
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
        embeds = self.word_embeddings(input_ids)
        if(utils.DROPOUT_EMBED > 0):
            embeds = self.dropout_embed(embeds)

        #embeds = torch.nn.utils.rnn.pack_padded_sequence(
        #    embeds, sentence[1], batch_first=True, enforce_sorted=False)

        lstm_out, _ = self.lstm(embeds)
        output_padded = utilz.get_senses_vector(lstm_out,idx,word_ids )

        #output_padded, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
        #    lstm_out, batch_first=True)

        if(utils.DROPOUT_LAYER > 0):
            output_padded = self.dropout_layer(output_padded)

        labels_space = self.hidden2labels(output_padded)
        return labels_space
    def configure_optimizers(self):
         groups = [
            {
                "params": self.hidden2labels.parameters(),
                "lr": utils.LEARNING_RATE,
                #"weight_decay": utils.weight_decay,
            },
            {
                "params": self.lstm.parameters(),
                "lr": utils.LEARNING_RATE,
                #"weight_decay": utils.transformer_weight_decay,
            }
         ]
    def training_step(self, train_batch,idx) -> STEP_OUTPUT:
        outputs = self(**train_batch)

        loss = F.cross_entropy(outputs.view(-1, self.num_labels),train_batch["labels"].view(-1),ignore_index=-100)
        self.log_dict({'train_loss':loss},batch_size=utilz.BATCH_SIZE,on_epoch=True, on_step=False,prog_bar=True)
        return loss
    
    def validation_step(self, val_batch,idx):
        outputs = self(**val_batch)
        #print(outputs.size())
        y_pred = outputs.argmax(dim = 1)
        #print(val_batch["labels"].size())
       
       
        loss = F.cross_entropy(outputs.view(-1, self.num_labels),val_batch["labels"].view(-1),ignore_index=-100)
        self.val_metric(y_pred,val_batch["labels"])
        self.log_dict({'val_loss':loss,'valid_f1': self.val_metric},batch_size=utilz.BATCH_SIZE,on_epoch=True, on_step=False,prog_bar=True)
    def test_step(self, test_batch,idx):
        
        outputs = self(**test_batch)
        y_pred = outputs.argmax(dim = 1)
        predicted_labels = utilz.idx_to_label(
                    self.label_list, y_pred.tolist())
        #print("RES: " + str(predicted_labels))
        #print(utilz.idx_to_label(
        #            self.label_list, test_batch["labels"].tolist()))
        loss = F.cross_entropy(outputs.view(-1, self.num_labels),test_batch["labels"].view(-1),ignore_index=-100)

        self.test_metric(y_pred,test_batch["labels"])
        self.log_dict({'test_loss':loss,'loss_f1': self.test_metric},batch_size=utilz.BATCH_SIZE,on_epoch=True, on_step=False,prog_bar=True)
        