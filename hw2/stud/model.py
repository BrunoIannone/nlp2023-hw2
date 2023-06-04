from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F
from transformers import AutoModel
import utilz
import os
import pytorch_lightning as pl
import time
import torchmetrics


class WSD(pl.LightningModule): #//TODO vedere se far brillare label_list
    def __init__(self, language_model_name: str, num_labels: int, label_list,fine_tune_lm: bool = True, *args, **kwargs) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.label_list = label_list
        # layers
        self.transformer_model = AutoModel.from_pretrained(language_model_name, output_hidden_states=True)
        if not fine_tune_lm:
            for param in self.transformer_model.parameters():
                param.requires_grad = False
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(
            self.transformer_model.config.hidden_size, num_labels, bias=True
        )
        self.val_metric = torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='macro')
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
        # group model inputs and pass to the model
        model_kwargs = {
          "input_ids": input_ids, 
          "attention_mask": attention_mask
        }
        
        # not every model supports token_type_ids
        if token_type_ids is not None:
          model_kwargs["token_type_ids"] = token_type_ids
        
        
        transformers_outputs = self.transformer_model(**model_kwargs)
        
        transformers_outputs = transformers_outputs[0]
        
        res = utilz.get_senses_vector(transformers_outputs,idx )
        
        logits = F.log_softmax(self.classifier(res),dim = 1)
                
        return logits

    
    def configure_optimizers(self):
            
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    def training_step(self,train_batch,batch_idx):

        outputs = self(**train_batch)
        
        
        loss = F.cross_entropy(outputs.view(-1, self.num_labels),train_batch["labels"].view(-1),ignore_index=-100)
        self.log('train_loss', loss,batch_size=utilz.BATCH_SIZE)
        return loss
    
    def validation_step(self, val_batch,idx):
        outputs = self(**val_batch)

       
        
        loss = F.cross_entropy(outputs.view(-1, self.num_labels),val_batch["labels"].view(-1),ignore_index=-100)
        self.log('val_loss', loss,batch_size=utilz.BATCH_SIZE) 
        self.log('f1', self.val_metric(outputs,val_batch["labels"]))
