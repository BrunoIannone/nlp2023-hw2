from torch.utils.data import Dataset
#import stud.utilz as utilz
import  transformer_utils

from typing import List
import time
class WsdDataset(Dataset):
    """WSD dataset class
    """
    def __init__(self, samples: List[dict], labels_to_idx: dict):
        """Constructor for the WSD dataset

        Args:
            samples (List[dict]): List of samples dict {"instance_ids": [instance_ids], "lemmas": [lemmas], "words": [words], "pos_tags": [pos_tags], "senses": [senses], "candidates":[candidates]}
            labels_to_idx (dict): dictionary with structure {label:index}
        """
        
        self.samples = samples 
        
        self.labels_to_idx = labels_to_idx
        
        
    

    def __len__(self):
        """Return samples length

        Returns:
            int: length of samples list (number of samples)_
        """
        return len(self.samples)

    def __getitem__(self, index: int):
        
        """Get item for dataloader input

        Args:
            index (int): index-th sample to access

        Returns:
            dict: {"sample": sample_dict} related to the index-th element with senses converted into their indices
        """
        
        #converte index-th sample senses in indices
        self.samples[index]["senses"] = transformer_utils.label_to_idx(self.labels_to_idx, self.samples[index]["senses"])
        return {
            
            "sample": self.samples[index],
            #"senses": utilz.label_to_idx(self.labels_to_idx, self.samples[index]["senses"])
    
        }
