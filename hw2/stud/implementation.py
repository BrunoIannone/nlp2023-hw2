import numpy as np
from typing import List, Dict
import stud.wsd_model as wsd_model
from model import Model
from transformers import AutoTokenizer
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
def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    x = "First commit"
    return StudentModel(device)

DIRECTORY_NAME = os.path.dirname(__file__)

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
        #print(self.vocab.keys())
        self.LANGUAGE_MODEL_NAME = 'kanishka/GlossBERT'
        self.device = device
        self.TOKENIZER = AutoTokenizer.from_pretrained(utilz.LANGUAGE_MODEL_PATH, use_fast=True,add_prefix_space = True)
        #self.model = wsd_model.WSD(LANGUAGE_MODEL_NAME,len(self.vocab.labels_to_idx.keys()),self.vocab.idx_to_labels, fine_tune_lm=True)
        self.model = wsd_model.WSD.load_from_checkpoint(os.path.join(DIRECTORY_NAME, 'glossbert2_0.884.ckpt'),map_location=self.device)
    
    def load_vocabularies(self, path: str,tokens_vocab = False):
        """_summary_

        Args:
            path (str): path to vocabularies word_to_idx and viceversa, labels_to_idx and viceversa

        Returns:
            dict: a dictionary containing the four others dictionaries: {"word_to_idx":word_to_idx,"idx_to_word":idx_to_word,"labels_to_idx":labels_to_idx,"idx_to_labels":idx_to_labels}
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
    
    def predict(self, sentences: List[Dict]) -> List[List[str]]:
        self.model.eval()
        #samples = []
        res = []
        for json_line in sentences:
            print(json_line)
            #samples = []
            predicted_labels = self.predict_(json_line)
            res.append(predicted_labels)
        return res
    
    def predict_(self,json_line):
        json_line["senses"] = utilz.label_to_idx(self.vocab["labels_to_idx"], json_line["candidates"])
        #samples.append({"instance_ids": json_line["instance_ids"], "lemmas": json_line["lemmas"], "words": json_line["words"],
        #            "pos_tags": json_line["pos_tags"], "senses": json_line["candidates"], "candidates": json_line["candidates"]})
        json_line = {"sample": json_line}
            
        batch = utilz.collate_fn([json_line]).to(self.device)
           
        y_pred = self.model(**batch)
        y_pred = y_pred.argmax(dim = 1)
        predicted_labels = utilz.idx_to_label(
        list(self.vocab["labels_to_idx"].keys()), y_pred.tolist())
        return predicted_labels
    
    def dict_to_dataset(self, batch):

        samples = []
        
        for json_line in batch:
            
            samples.append({"instance_ids": json_line["instance_ids"], "lemmas": json_line["lemmas"], "words": json_line["words"],
                        "pos_tags": json_line["pos_tags"], "senses": json_line["candidates"], "candidates": json_line["candidates"]})
        

        return {
            "samples": samples
        }
        
   