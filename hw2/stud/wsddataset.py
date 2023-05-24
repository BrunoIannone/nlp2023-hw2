from torch.utils.data import Dataset
import utils
from typing import List
import torch 
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
import time
class WsdDataset(Dataset):
    """WSD dataset class
    """
    def __init__(self, sentences: List[List[str]], labels: List[List[str]], word_to_idx: dict, labels_to_idx: dict):
        """Constructor for the WSD dataset

        Args:
            sentences (List[List[str]]): List of list of sentence tokens
            labels (List[List[str]]): List of list of sentences token labels
            word_to_idx (dict): dictionary with structure {word:index}
            labels_to_idx (dict): dictionary with structure {label:index}
        """
        self.sentences = sentences
        self.labels = labels
        self.samples = self._preprocess_samples(sentences, labels)
        self.word_to_idx = word_to_idx
        self.labels_to_idx = labels_to_idx

    # notebook 3
    def _preprocess_samples(self, sentences: List[List[str]], labels: List[List[str]]):
        """Aux function for samples preprocessing

        Args:
            sentences (List[List[str]]): List of sentence tokens
            labels (List[List[str]]): List of sentences token labels

        Returns:
            List[tuple]: List of tuples (List[token indexes (int)], List[labels indexes (int)])
        """
        res = []
        for sentence, label in zip(sentences, labels):
            res.append((sentence, label))
        #print(res)
        return res

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        """Get item for dataloader input

        Args:
            index (int): index-th sample to access

        Returns:
            tuple: (List[token indexes (int)], List[labels indexes (int)]) of the index-th element
        """
        sentence = self.samples[index][0]
        #print(sentence)

        labels = self.samples[index][1]
        length = len(sentence)
        #print(length)
        
        temp = ['O'] * length
        for sense in labels:
            #print(sense)
            #print(temp)
            temp[int(sense)] = labels[sense][0]
        #print(temp)
        #print(len(temp))
        #res = utils.word_to_idx(self.word_to_idx, sentence), utils.label_to_idx(
            #self.labels_to_idx, labels)
        #print(temp)
        #time.sleep(5)
        return {
            'tokens': sentence,
            'ner_tags': utils.label_to_idx(self.labels_to_idx, temp)
        }
