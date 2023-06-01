from torch.utils.data import Dataset
import utilz
from typing import List
import torch 
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
import time
class WsdDataset(Dataset):
    """WSD dataset class
    """
    def __init__(self, samples, word_to_idx: dict, labels_to_idx: dict):
        """Constructor for the WSD dataset

        Args:
            sentences (List[List[str]]): List of list of sentence tokens
            labels (List[List[str]]): List of list of sentences token labels
            word_to_idx (dict): dictionary with structure {word:index}
            labels_to_idx (dict): dictionary with structure {label:index}
        """
        #self.sentences = sentences
        #self.labels = labels
        self.samples = samples #self._preprocess_samples(sentences, labels,instance_ids)
        self.word_to_idx = word_to_idx
        self.labels_to_idx = labels_to_idx
        
        #self.instance_ids = instance_ids
    # notebook 3
    def _preprocess_samples(self, sentences: List[List[str]], labels: List[List[str]],ids):
        """Aux function for samples preprocessing

        Args:
            sentences (List[List[str]]): List of sentence tokens
            labels (List[List[str]]): List of sentences token labels

        Returns:
            List[tuple]: List of tuples (List[token indexes (int)], List[labels indexes (int)])
        """
        res = []
        for sentence, label,id in zip(sentences, labels,ids):
            #print(sentence, label,id)
            #time.sleep(10)
            
            res.append({"words":sentence,"label": label,"id": id})
            #print(res)
            #time.sleep(10)
        return res

    def __len__(self):
        #print(self.samples)
        return len(self.samples)

    def __getitem__(self, index: int):
        #print(index)
        #print(self.samples[0:5])
        #time.sleep(100)
        """Get item for dataloader input

        Args:
            index (int): index-th sample to access

        Returns:
            tuple: (List[token indexes (int)], List[labels indexes (int)]) of the index-th element
        """
        """ id = self.samples[index]["id"]
        print(id)
        sentence = self.samples[index]["words"]
        print(sentence)
        labels = self.samples[index]["label"]

        print(labels)
        print("OPATCHKI")  
        #time.sleep(5)
 """
        #print(self.samples[index])
        #time.sleep(5)
        return {
            
            "sample": self.samples[index],
            "senses": utilz.label_to_idx(self.labels_to_idx, self.samples[index]["senses"])
    
        }
            #'tokens': self.samples[index]["words"],
            #'ner_tags': utils.label_to_idx(self.labels_to_idx, self.samples[index]["label"])
    
