from typing import Any, Optional
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import torch
import torch.nn.functional as F
from transformers import AutoModel, BertForTokenClassification,AutoModelForTokenClassification,optimization
#import stud.utilz as utilz
import transformer_utils
import pytorch_lightning as pl
import time
import torchmetrics
import numpy as np
import wsd_model
import os
class WSD(pl.LightningModule): 
    def __init__(self, language_model_name: str, num_coarse_labels: int, num_fine_labels,idx_to_labels:dict,labels_to_idx:dict,coarse_to_fine,fine_tune_lm: bool = True,lin_lr = 0, backbone_lr = 0,lin_wd = 0, backbone_wd = 0, lin_dropout = 0 ,*args, **kwargs) -> None:
        super().__init__()
        print(num_coarse_labels,num_fine_labels)
        self.num_fine_labels = num_fine_labels
        self.idx_to_labels = idx_to_labels
        self.labels_to_idx = labels_to_idx
        self.lin_wd = lin_wd
        self.backbone_wd = backbone_wd
        self.lin_lr = lin_lr
        self.backbone_lr = backbone_lr
        self.coarse_to_fine = coarse_to_fine
        self.relu = torch.nn.LeakyReLU()
        #self.classifier2 = torch.nn.Linear(num_coarse_labels, num_fine_labels)
        #self.lin_dropout = lin_dropout
        #print("SI")
        #self.backbone = AutoModel.from_pretrained(language_model_name,output_hidden_states=True,num_labels = num_coarse_labels)
        self.backbone = wsd_model.WSD.load_from_checkpoint(os.path.join(transformer_utils.DIRECTORY_NAME, '../../model/0.001, 0.002, 0.01.ckpt'),map_location='cpu', strict=False)
        #print("SI2initializing")
        #self.transformer_pos_model = AutoModel.from_pretrained(language_model_name_pos,output_hidden_states=True,num_labels = num_labels)
        #for param in self.transformer_pos_model.parameters():
        #        param.requires_grad = False
        
        if not fine_tune_lm:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.lin_dropout = torch.nn.Dropout(lin_dropout)
        #self.classifier = torch.nn.Linear(
        #    self.backbone.config.hidden_size, num_coarse_labels, bias=True
        #)
        
        
        self.val_metric  = torchmetrics.F1Score(task="multiclass", num_classes=num_fine_labels, average='micro')
        self.test_metric = torchmetrics.F1Score(task="multiclass", num_classes=num_fine_labels, average='micro')

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
          "attention_mask": attention_mask,
          "idx": idx,
          "word_ids":word_ids
        }
        
        # not every model supports token_type_ids
        if token_type_ids is not None:
          model_kwargs["token_type_ids"] = token_type_ids
        
        return self.backbone(**model_kwargs)
        # transformers_outputs = self.backbone(**model_kwargs)    
        # transformers_outputs_sum = torch.stack(transformers_outputs.hidden_states[-4:], dim=0).sum(dim=0)
        
        # transformers_outputs_sum = utilz.get_senses_vector(transformers_outputs_sum,idx,word_ids )
        # logits = self.lin_dropout(transformers_outputs_sum)
        # #logits = self.relu(transformers_outputs)
        # #print(logits.size())
        # #logits = self.lin_dropout(logits)
        
        # logits = self.classifier(logits)   
        # #print(logits.size())
        # y_pred = logits.argmax(dim = 1)
        res = []
        max_similarity = -2.0
        max_fine_key = None
        for i in range(len(y_pred)):
            #print(self.coarse_to_fine)
            #time.sleep(5)

            coarse = self.coarse_to_fine[self.idx_to_labels[int(y_pred[i])]] # {fine_key:gloss}
            cos = torch.nn.CosineSimilarity(dim=0)
            #print(coarse)
            #time.sleep(5)

            for dict in coarse:
                for fine_key in dict:
                    #print(fine_key)
                    #print(dict[fine_key])
                    #time.sleep(5)
                    #print(transformers_outputs_sum[i].size())
                    #print(dict[fine_key].size())
                    #time.sleep(5)

                    cosine_similarity = cos(transformers_outputs_sum[i],dict[fine_key])
                    
                    print(cosine_similarity)

                    if cosine_similarity > max_similarity:
                        print("SI", cosine_similarity)
                        max_similarity = cosine_similarity
                        max_fine_key = fine_key
            #print("finekeymax",max_fine_key)
            #print("ESO",self.labels_to_idx[max_fine_key])   
            res.append(torch.tensor(self.labels_to_idx[max_fine_key]))
            #print(res)
            #time.sleep(5)
            max_similarity = -1.0
            max_fine_key = None    
                #print(res)
                #time.sleep(5)
            

        res = torch.stack(res, dim=0)
        return res

    
    def configure_optimizers(self):
        
        groups = [
          #{
           #   "params": self.classifier.parameters(),
           #   "lr": self.lin_lr,
           #   "weight_decay": self.lin_wd,
           # },
            # {
            #    "params": self.classifier2.parameters(),
            #    "lr": self.lin_lr,
            #    "weight_decay": self.lin_wd,
            # },
           {
               "params": self.backbone.parameters(),
               "lr": self.backbone_lr,
               "weight_decay": self.backbone_wd,
           }
            
             
        ]           
        
        optimizer = torch.optim.AdamW(groups)
        
        
        #print("FINE OPTIMIZWE")
        return optimizer
    
    def training_step(self,train_batch,batch_idx):

        outputs = self(**train_batch)
        
        loss = F.cross_entropy(outputs.view(-1, self.num_fine_labels),train_batch["labels"].view(-1),ignore_index=-100)
        
        self.log_dict({'train_loss':loss},on_epoch=True, batch_size=transformer_utils.BATCH_SIZE,on_step=False,prog_bar=True)
        
        return loss
        

    def validation_step(self, val_batch,idx):
        outputs = self(**val_batch)
        y_pred = outputs.argmax(dim = 1)
       
       
        loss = F.cross_entropy(outputs.view(-1, self.num_fine_labels),val_batch["labels"].view(-1),ignore_index=-100)
        #print(val_batch["labels"])
        self.val_metric(y_pred,val_batch["labels"])
        self.log_dict({'val_loss':loss,'valid_f1': self.val_metric},batch_size=transformer_utils.BATCH_SIZE,on_epoch=True, on_step=False,prog_bar=True)
    
        
    def test_step(self, test_batch,idx):
        
        outputs = self(**test_batch)
        y_pred = outputs[0].argmax(dim = 1)
        res = []
        max_similarity = -2.0
        max_fine_key = None
        for i in range(len(y_pred)):
            #print(self.coarse_to_fine)
            #time.sleep(5)

            coarse = self.coarse_to_fine[self.idx_to_labels[int(y_pred[i])]] # {fine_key:gloss}
            cos = torch.nn.CosineSimilarity(dim=0)
            #print(coarse)
            #time.sleep(5)

            for dict in coarse:
                for fine_key in dict:
                    #print(fine_key)
                    #print(dict[fine_key])
                    #time.sleep(5)
                    #print(transformers_outputs_sum[i].size())
                    #print(dict[fine_key].size())
                    #time.sleep(5)

                    #cosine_similarity = cos(torch.nn.functional.normalize(outputs[1][i],dim = 0),torch.nn.functional.normalize(dict[fine_key],dim=0).to("cuda"))
                    cosine_similarity = torch.dot(torch.nn.functional.normalize(outputs[1][i],dim = 0),torch.nn.functional.normalize(dict[fine_key],dim=0).to("cuda"))
                    print(cosine_similarity)

                    if cosine_similarity > max_similarity:
                        #print("SI", cosine_similarity)
                        max_similarity = cosine_similarity
                        max_fine_key = fine_key
            #print("finekeymax",max_fine_key)
            #print("ESO",self.labels_to_idx[max_fine_key])   
            res.append(torch.tensor(self.labels_to_idx[max_fine_key]))
            #print(res)
            #time.sleep(5)
            max_similarity = -1.0
            max_fine_key = None    
                #print(res)
                #time.sleep(5)
            

        res = torch.stack(res, dim=0).to('cuda')
        #return res
        
        #loss = F.cross_entropy(outputs.view(-1, self.num_fine_labels),test_batch["labels"].view(-1),ignore_index=-100)
        loss = 0
        #self.test_metric(y_pred,test_batch["labels"])
        self.test_metric(res,test_batch["labels"])
        
        self.log_dict({'test_loss':loss,'test_f1': self.test_metric},batch_size=transformer_utils.BATCH_SIZE,on_epoch=True, on_step=False,prog_bar=True)
                      

    
    