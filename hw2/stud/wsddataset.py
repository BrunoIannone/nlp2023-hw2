from torch.utils.data import Dataset
import stud.utilz as utilz
#import  utilz

from typing import List
import torch 
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
import time
class WsdDataset(Dataset):
    """WSD dataset class
    """
    def __init__(self, samples, labels_to_idx: dict):
        """Constructor for the WSD dataset

        Args:
            sentences (List[List[str]]): List of list of sentence tokens
            labels (List[List[str]]): List of list of sentences token labels
            labels_to_idx (dict): dictionary with structure {label:index}
        """
        
        self.samples = samples 
        
        self.labels_to_idx = labels_to_idx
        
        
    

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        
        """Get item for dataloader input

        Args:
            index (int): index-th sample to access

        Returns:
            dict: {"sample": sample_dict, List[sense indexes (int)]} related to the index-th element
        """
        
        
        return {
            
            "sample": self.samples[index],
            "senses": utilz.label_to_idx(self.labels_to_idx, self.samples[index]["senses"])
    
        }
           
