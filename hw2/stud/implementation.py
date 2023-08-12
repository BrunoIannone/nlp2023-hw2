import numpy as np
from typing import List, Dict
import stud.wsd_model as wsd_model
from model import Model
from transformers import AutoTokenizer,AutoModel
import os
import stud.utilz as utilz
import json
import pytorch_lightning as pl
from typing import List
import stud.wsddataset as wsddataset
from torch.utils.data import DataLoader
import torch
import os
import time
import torch.nn.functional as F

def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    x = "First commit"
    return StudentModel(device)

DIRECTORY_NAME = os.path.dirname(__file__)
print(os.path.join(DIRECTORY_NAME, '../../model/GlossBERT/'))
class RandomBaseline(Model):

    def __init__(self):
        # Load your models/tokenizer/etc. that only needs to be loaded once when doing inference
        pass

    def predict(self, sentences: List[Dict]) -> List[List[str]]:
        return [[np.random.choice(candidates) for candidates in sentence_data["candidates"].values()]
                for sentence_data in sentences]


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self,device):
        # Load your models/tokenizer/etc. that only needs to be loaded once when doing inference
        self.vocab = self.load_vocabularies(DIRECTORY_NAME)
        self.LANGUAGE_MODEL_NAME = 'kanishka/GlossBERT'
        self.device = device
        self.model = wsd_model.WSD.load_from_checkpoint(os.path.join(DIRECTORY_NAME,'../../model/0.001, 0.002, 0.01.ckpt'),map_location=self.device)
        self.model.eval()

    def load_vocabularies(self, path: str,tokens_vocab = False):
        """Load vocabularies

        Args:
            path (str): path to vocabularies word_to_idx and viceversa, labels_to_idx and viceversa
            tokens_vocab(bool): If True, loads word_to_idx and idx_to_word vocabularies
        Returns:
            if(tokens_vocab == True)
                dict: a dictionary containing the four others dictionaries: {"word_to_idx":word_to_idx,"idx_to_word":idx_to_word,"labels_to_idx":labels_to_idx,"idx_to_labels":idx_to_labels}
            else:
                {"labels_to_idx":labels_to_idx,"idx_to_labels":idx_to_labels}
            
                
        """
        vocab = {}
        if(tokens_vocab):
            with open(os.path.join(path, "../../model/word_to_idx.txt"), "r") as fp:
                vocab["word_to_idx"] = json.load(fp)
                fp.close()

            with open(os.path.join(path, "../../model/idx_to_word.txt"), "r") as fp:
                vocab["idx_to_word"] = json.load(fp, object_hook=utilz.str_to_int)
                fp.close()

        with open(os.path.join(path, "../../model/labels_to_idx.txt"), "r") as fp:
            vocab["labels_to_idx"] = json.load(fp)
            fp.close()

        with open(os.path.join(path,  "../../model/idx_to_labels.txt"), "r") as fp:
            vocab["idx_to_labels"] = json.load(
                fp, object_hook=utilz.str_to_int)
            fp.close()
        return vocab
    
    def predict(self, samples: List[Dict]) -> List[List[str]]:
        self.model.eval()
        
        res = []
        for sample in samples:
            
            predicted_labels = self.predict_(sample)
            res.append(predicted_labels)

        return res
    
    def predict_(self,sample): #TODO: documentation missing
        
        unknown_candidates = []
        modified = False

        for candidate in sample["candidates"]:
            if len(sample["candidates"][candidate]) == 1 and sample["candidates"][candidate][0] in self.vocab["labels_to_idx"].keys():
            
                unknown_candidates.append(0)
            
            elif len(sample["candidates"][candidate]) == 1 and sample["candidates"][candidate][0] not in self.vocab["labels_to_idx"].keys():

                unknown_candidates.append(sample["candidates"][candidate][0])
            
            else: 
                
                for candidate_value in sample["candidates"][candidate]:
                    
                    if candidate_value in self.vocab["labels_to_idx"].keys():
                    
                        unknown_candidates.append(0)
                    
                        modified = True
                        break
                
                if not modified:
                    unknown_candidates.append(sample["candidates"][candidate][0])
            
        sample["senses"] = utilz.label_to_idx(self.vocab["labels_to_idx"], sample["candidates"])
        sample = {"sample": sample}
            
        batch = utilz.collate_fn([sample]).to(self.device)
           
        y_pred = self.model(**batch)
        y_pred = y_pred.argmax(dim = 1)

        predicted_labels = utilz.idx_to_label(
        list(self.vocab["labels_to_idx"].keys()), y_pred.tolist())
        
        for i in range(len(unknown_candidates)):
            if unknown_candidates[i] == 0:
                continue
            else:
                predicted_labels[i] = unknown_candidates[i]
        
        
        return predicted_labels
    
    
    
    def max_result(self,y_pred,known_candidate):
        max = -1
        res = None
        print(known_candidate)
        #time.sleep(5)
        for candidate in known_candidate:
            if y_pred[0][candidate] > max and y_pred[0][candidate] > 0.51 :
                res = candidate
                max = y_pred[0][candidate]
        return res


    
    def predict__(self,sample):
        res = []
        for target in sample["candidates"]:
            #print(sample["candidates"][target])
            if len(sample["candidates"][target]) == 1:
                res.append(sample["candidates"][target][0])
            elif all(candidate in self.vocab["labels_to_idx"].keys() for candidate in sample["candidates"][target]):
                
                
                sample["senses"] = utilz.label_to_idx(self.vocab["labels_to_idx"],  {target : sample["candidates"][target]})
                sample_ = {"sample": sample}
            
                batch = utilz.collate_fn([sample_]).to(self.device)
           
                y_pred = self.model(**batch)
                print(sample["senses"])
                time.sleep(5)
                y_pred_temp = self.max_result(y_pred,[sample["senses"][target]])
                if y_pred_temp == None:
                    y_pred = y_pred.argmax(dim = 1)
                    predicted_labels = utilz.idx_to_label(
                    list(self.vocab["labels_to_idx"].keys()), [y_pred])
                    res.append(predicted_labels)
                else:
                    predicted_labels = utilz.idx_to_label(
                    list(self.vocab["labels_to_idx"].keys()), [y_pred_temp])
                    res.append(predicted_labels)
            else:
                known_candidate = []
                unknown_candidate = []
                for candidate in sample["candidates"][target]:
                    #print(candidate)
                    if candidate in self.vocab["labels_to_idx"].keys():
                        known_candidate.append(self.vocab["labels_to_idx"][candidate])
                    else:
                        #print(target,candidate)

                        unknown_candidate.append(candidate)
                if(known_candidate == []):
                    res.append(unknown_candidate[0])
                    continue
                #print(sample["candidates"][target])
                sample["senses"] = utilz.label_to_idx(self.vocab["labels_to_idx"], {target : sample["candidates"][target]})
                sample_ = {"sample": sample}
            
                batch = utilz.collate_fn([sample_]).to(self.device)
           
                y_pred = F.softmax(self.model(**batch),dim = 1)
                #print(y_pred,y_pred.size())
                y_pred_temp = self.max_result(y_pred,known_candidate)
                if y_pred_temp == None:
                    res.append(unknown_candidate[0])
                else:
                    predicted_labels = utilz.idx_to_label(
                    list(self.vocab["labels_to_idx"].keys()), [y_pred_temp])
                    res.append(predicted_labels)

        return res