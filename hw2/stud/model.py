from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F
from transformers import AutoModel
import utilz
import os
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import time
import torchmetrics


class WSD(pl.LightningModule):
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
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='macro')
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
        
        #idx = utilz.get_idx_from_tensor(idx)
            
        
        #print(input_ids)
        #print(idx)
        #print(input_ids.size()) #batch x max_length
        #print(input_ids)
        #print(labels.size()) #batch x 1
        #time.sleep(5)
        transformers_outputs = self.transformer_model(**model_kwargs)
        #print("TRANSFORMERS_OUTPUTS: " + str(transformers_outputs)) #128 x max_length x 768 (hidden_size) 
        x = transformers_outputs[0]
        # we would like to use the sum of the last four hidden layers
        #transformers_outputs_sum = torch.stack(transformers_outputs.hidden_states[-4:], dim=0).sum(dim=0)
        #transformers_outputs_sum = self.dropout(transformers_outputs_sum)
        #print("transformers_outputs_sum[0]: " + str(transformers_outputs_sum[0].size()))
        #logits = F.log_softmax(self.classifier(transformers_outputs_sum),dim = -1)
        
        #print("OUTPUT: " + str(transformers_outputs["hidden_states"]))
        res = utilz.get_senses_vector(x,idx )
        
        logits = F.log_softmax(self.classifier(res),dim = 1)
        #print("LOGITS: " + str(logits.size()))
                
        return logits

    
    def configure_optimizers(self):
            
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    def training_step(self,train_batch,batch_idx):
        outputs = self(**train_batch)
        #print(outputs.size())
        #print(train_batch["labels"].size())
        
        loss = F.cross_entropy(outputs.view(-1, self.num_labels),train_batch["labels"].view(-1),ignore_index=-100)
        self.log('train_loss', loss,batch_size=utilz.BATCH)
        return loss
    
    def validation_step(self, val_batch,idx):
        outputs = self(**val_batch)
        #print(outputs.size())
        #print(val_batch["labels"].size())
        
        loss = F.cross_entropy(outputs.view(-1, self.num_labels),val_batch["labels"].view(-1),ignore_index=-100)
        self.log('val_loss', loss,batch_size=utilz.BATCH) 
        self.log('f1', self.f1(outputs,val_batch["labels"]))
        #print(utilz.compute_metrics(val_batch["labels"],outputs,self.label_list))