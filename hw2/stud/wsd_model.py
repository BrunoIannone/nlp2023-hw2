from typing import Any, Optional
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import torch
import torch.nn.functional as F
from transformers import AutoModel, BertForTokenClassification,AutoModelForTokenClassification,optimization

#import stud.utilz as utilz
import utilz
import pytorch_lightning as pl
import time
import torchmetrics

class WSD(pl.LightningModule): 
    def __init__(self, language_model_name: str, num_labels: int, idx_to_labels:dict,fine_tune_lm: bool = True,*args, **kwargs) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.idx_to_labels = idx_to_labels
        
        
        self.learning_rate = 1e-3
        self.backbone = AutoModel.from_pretrained(language_model_name, output_hidden_states=True,num_labels = num_labels)
        #self.transformer_pos_model = AutoModel.from_pretrained(language_model_name_pos,output_hidden_states=True,num_labels = num_labels)
        #for param in self.transformer_pos_model.parameters():
        #        param.requires_grad = False
        
        if not fine_tune_lm:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.dropout = torch.nn.Dropout(0.8)
        self.classifier = torch.nn.Linear(
            self.backbone.config.hidden_size, num_labels, bias=True
        )
        
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
        
        
        transformers_outputs = self.backbone(**model_kwargs)    
        transformers_outputs_sum = torch.stack(transformers_outputs.hidden_states[-4:], dim=0).sum(dim=0)
        transformers_outputs_sum = utilz.get_senses_vector(transformers_outputs_sum,idx,word_ids )
        transformers_outputs_sum = self.dropout(transformers_outputs_sum)
        logits = self.classifier(transformers_outputs_sum)   
        return logits

    
    def configure_optimizers(self):
        
        groups = [
          {
               "params": self.classifier.parameters(),
               "lr": self.learning_rate,
               "weight_decay": utilz.weight_decay,
            },
        #    {
        #        "params": self.backbone.parameters(),
        #        "lr": utilz.transformer_learning_rate,
        #        "weight_decay": utilz.transformer_weight_decay,
        #    }
            
             
        ]           
        #optimizer =optimization.Adafactor(groups)
        optimizer =torch.optim.AdamW(groups)

        
        
        
        return [optimizer]
    
    def training_step(self,train_batch,batch_idx):

        outputs = self(**train_batch)
        
        loss = F.cross_entropy(outputs.view(-1, self.num_labels),train_batch["labels"].view(-1),ignore_index=-100)
        
        self.log_dict({'train_loss':loss},on_epoch=True, batch_size=utilz.BATCH_SIZE,on_step=False,prog_bar=True)
        return loss
        

    def validation_step(self, val_batch,idx):
        outputs = self(**val_batch)
        y_pred = outputs.argmax(dim = 1)
       
       
        loss = F.cross_entropy(outputs.view(-1, self.num_labels),val_batch["labels"].view(-1),ignore_index=-100)
        self.val_metric(y_pred,val_batch["labels"])
        self.log_dict({'val_loss':loss,'valid_f1': self.val_metric},batch_size=utilz.BATCH_SIZE,on_epoch=True, on_step=False,prog_bar=True)
    
        
    def test_step(self, test_batch,idx):
        
        outputs = self(**test_batch)
        y_pred = outputs.argmax(dim = 1)
        
        loss = F.cross_entropy(outputs.view(-1, self.num_labels),test_batch["labels"].view(-1),ignore_index=-100)

        self.test_metric(y_pred,test_batch["labels"])
        self.log_dict({'test_loss':loss,'test_f1': self.test_metric},batch_size=utilz.BATCH_SIZE,on_epoch=True, on_step=False,prog_bar=True)
                      

    
    