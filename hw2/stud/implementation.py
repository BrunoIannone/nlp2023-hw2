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

def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    x = "First commit"
    return StudentModel()

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

    def __init__(self):
        # Load your models/tokenizer/etc. that only needs to be loaded once when doing inference
        self.vocab = self.load_vocabularies(DIRECTORY_NAME)
        print(self.vocab.keys())
        self.LANGUAGE_MODEL_NAME = 'kanishka/GlossBERT'
        
        self.TOKENIZER = AutoTokenizer.from_pretrained(self.LANGUAGE_MODEL_NAME, use_fast=True,add_prefix_space = True)
        #self.model = wsd_model.WSD(LANGUAGE_MODEL_NAME,len(self.vocab.labels_to_idx.keys()),self.vocab.idx_to_labels, fine_tune_lm=True)
        self.model = wsd_model.WSD.load_from_checkpoint(os.path.join(
            DIRECTORY_NAME, 'glossbert2_0.884.ckpt'))
    
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
        print(sentences)
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        res = []
        trainer = pl.Trainer()

        #for elem in sentences:
            #elem = {"sample":{"words":elem["words"]},"senses": elem["senses"]}           
            #print(elem["sample"]["words"])
            ##test_data = utilz.collate_fn([elem])
            #res.extend(trainer.predict(self.model,test_data))
        test_data = self.dict_to_dataset(sentences)
        
        
        #elem = self.dict_to_dataset(sentences)
        #elem = elem["samples"]
        test_dataset = wsddataset.WsdDataset(test_data["samples"],self.vocab["labels_to_idx"])

        #print(elem)

        #elem = [{"sample":{"words":elem["words"]},"senses": utilz.label_to_idx(self.vocab["labels_to_idx"],elem["senses"])}]
        test =  DataLoader(
            test_dataset,
            batch_size = utilz.BATCH_SIZE,
            num_workers = 1,#utilz.NUM_WORKERS,
            shuffle = False,
            collate_fn=utilz.collate_fn
        )
        #trainer.predict(self.model,dataloaders=test)
        res = trainer.predict(self.model,test)
        return res

    def dict_to_dataset(self, batch):

        words = []
        samples = []
        labels = []
        
        for json_line in batch:
            if( "senses" in list(json_line.keys())):
                samples.append({"instance_ids": json_line["instance_ids"], "lemmas": json_line["lemmas"], "words": json_line["words"],
                            "pos_tags": json_line["pos_tags"], "senses": json_line["senses"], "candidates": json_line["candidates"]})
            else: 
                samples.append({"instance_ids": json_line["instance_ids"], "lemmas": json_line["lemmas"], "words": json_line["words"],
                            "pos_tags": json_line["pos_tags"], "senses": [{-1:["<pad>"]}], "candidates": json_line["candidates"]})


        return {
            "samples": samples,
            #"words": words,
            #"labels":labels
        }
        
    def collate_fn(self,batch):
    
        print(batch)
        #time.sleep(5)
        batch_out = self.TOKENIZER(
            [sentence["sample"]["words"] for sentence in batch],
            return_tensors="pt",
            padding=True,
            # We use this argument because the texts in our dataset are lists of words.
            is_split_into_words=True,
            truncation=True

        )
        word_ids = []
        for idx,sentence in enumerate(batch):
            word_ids.append(batch_out.word_ids(batch_index=idx))
        print(word_ids)
        last_index = None
        res = []
        i = 0
        for l in word_ids:
            i = 0
            temp = []
            last_index = None
            while(i<len(l)):
                if last_index != None and l[last_index] == l[i]:
                    i +=1

                    continue
                else:
                    temp.append(i)
                    last_index = i
                    i+=1 
            res.append(torch.tensor(temp))   

        word_ids_ = torch.nn.utils.rnn.pad_sequence(
            res, batch_first=True, padding_value=-1)
        labels, idx = utilz.extract_labels_and_sense_indices(batch)

        batch_out["word_ids"] = word_ids_
        batch_out["labels"] = labels
        batch_out["idx"] = idx

        return batch_out
