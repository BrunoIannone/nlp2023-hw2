from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import torch
import torch.nn.functional as F
from transformers import AutoModel, BertForTokenClassification,AutoModelForTokenClassification,optimization
#import stud.transformer_utils as transformer_utils
import transformer_utils
import pytorch_lightning as pl
import time
import torchmetrics

class WSD(pl.LightningModule): 
    def __init__(self, language_model_name: str, num_labels: int,fine_tune_lm: bool = True,lin_lr: float = 0.0, backbone_lr: float = 0.0,lin_wd: float = 0.0, backbone_wd: float = 0.0, lin_dropout: float = 0.0 ,*args, **kwargs) -> None:
        """WSD init function for transformer-based models

    Args:
        language_model_name (str): Transformer Hugging Face model name
        num_labels (int): Number of total labels 
        fine_tune_lm (bool, optional): If false, transformer parameters won't be updated. Defaults to True.
        lin_lr (float, optional): Linear layer learning rate. Defaults to 0.0.
        backbone_lr (float, optional): Transformer learning rate. Defaults to 0.0.
        lin_wd (float, optional): Linear layer weight decay. Defaults to 0.0.
        backbone_wd (float, optional): Transformer weight decay. Defaults to 0.0.
        lin_dropout (float, optional): Linear layer dropout. Defaults to 0.0.
    
    """
        super().__init__()
        
        self.num_labels = num_labels
        self.lin_wd = lin_wd
        self.backbone_wd = backbone_wd
        self.lin_lr = lin_lr
        self.backbone_lr = backbone_lr
        self.lin_dropout = torch.nn.Dropout(lin_dropout)
        self.backbone = AutoModel.from_pretrained(language_model_name, output_hidden_states=True,num_labels = num_labels)
        self.classifier = torch.nn.Linear(
            self.backbone.config.hidden_size, num_labels, bias=True
        )
        
        if not fine_tune_lm:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.val_metric  = torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='micro')
        self.test_metric = torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average='micro')

        self.save_hyperparameters()

    def forward(
        self,
        idx,
        word_ids,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Model forward pass

        Args:

            idx (Tensor): Target word indices
            word_ids (Tensor): Transformer word ids
            input_ids (Tensor): Transformer input ids
            attention_mask (Tensor): Transformer attention mask
            token_type_ids (Tensor, optional): Transformer token type ids (if any)

        Returns:
           Tensor: Model predictions
        """
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
        
        target_vector = transformer_utils.get_target_vector(transformers_outputs_sum,idx,word_ids )
        target_vector = self.lin_dropout(target_vector)
        
        logits = self.classifier(target_vector)   
        return logits

    def configure_optimizers(self):
        
        groups = [
          {
               "params": self.classifier.parameters(),
               "lr": self.lin_lr,
               "weight_decay": self.lin_wd,
            },
           {
               "params": self.backbone.parameters(),
               "lr": self.backbone_lr,
               "weight_decay": self.backbone_wd,
           } 
        ]           
        
        optimizer = torch.optim.AdamW(groups)
               
        return optimizer
    
    def training_step(self,train_batch,batch_idx):

        outputs = self(**train_batch)
        
        loss = F.cross_entropy(outputs.view(-1, self.num_labels),train_batch["labels"].view(-1),ignore_index=-100)
        
        self.log_dict({'train_loss':loss},on_epoch=True, batch_size=transformer_utils.BATCH_SIZE,on_step=False,prog_bar=True)
        
        return loss
        

    def validation_step(self, val_batch,idx):
        
        outputs = self(**val_batch)
        y_pred = outputs.argmax(dim = 1)
       
       
        loss = F.cross_entropy(outputs.view(-1, self.num_labels),val_batch["labels"].view(-1),ignore_index=-100)
        
        self.val_metric(y_pred,val_batch["labels"])
        self.log_dict({'val_loss':loss,'valid_f1': self.val_metric},batch_size=transformer_utils.BATCH_SIZE,on_epoch=True, on_step=False,prog_bar=True)
    
        
    def test_step(self, test_batch,idx):
        
        outputs = self(**test_batch)
        y_pred = outputs.argmax(dim = 1)
        
        loss = F.cross_entropy(outputs.view(-1, self.num_labels),test_batch["labels"].view(-1),ignore_index=-100)

        self.test_metric(y_pred,test_batch["labels"])
        self.log_dict({'test_loss':loss,'test_f1': self.test_metric},batch_size=transformer_utils.BATCH_SIZE,on_epoch=True, on_step=False,prog_bar=True)
                      

    
    