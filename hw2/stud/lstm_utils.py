import os
import json
#import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List

###HYPERPARAMETERS###
EMBEDDING_DIM = 25
LAYERS_NUM = 2
HIDDEN_DIM = 150
EPOCHS_NUM = 500
LEARNING_RATE = 0.01
CHANCES = 5
DROPOUT_LAYER = 0.2
DROPOUT_EMBED = 0.5
DROPOUT_LSTM = 0.8
BATCH_SIZE = 4096 #2^12
#####################
EARLY_STOP = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BIDIRECTIONAL = True
DIRECTORY_NAME = os.path.dirname(__file__)





def build_data_from_jsonl(file_path:str): 
    """Split the JSONL file in file_path in sentences and relative labels 

    Args:
        file_path (string): path to JSONL file

    Returns:
        dictionary: return a dictionary with keys "sentences" and "labels" having as value list of list of strings: {sentences: List[list[sentences]], labels: List[List[labels]]}
    """
    try:
        f = open(file_path, 'r')
    except OSError:
        print("Unable to open file in "+ str(file_path))
    line = f.readline()
    sentences = []
    labels = []
    while (line):

        json_line = json.loads(line)
        sentences.append(json_line['tokens'])
        labels.append(json_line['labels'])
        line = f.readline()

    f.close()

    return {
        'sentences': sentences,
        'labels': labels

    }
    

def label_to_idx(labels_to_idx:dict, labels: List[str]):
    """Converts labels string in integer indexes. 
       

    Args:
        labels_to_idx (dictionary): dictionary with structure {label:index} 
        labels (List[string]): List of labels (stings)

    Returns:
        list: list of integers that represent labels indexes
    """
    res = []
    for label in labels:
        res.append(labels_to_idx[label])
    return res

def word_to_idx(word_to_idx:dict,sentence:List[str]):
    """Converts tokens of strings in their indexes. If a token is unknown, its index is the <unk> key value

    Args:
        word_to_idx (dict): dictionary with structure {word:index}
        sentence (list): list of tokens (strings)

    Returns:
        list: list of integers that represent tokens indexes
    """
    
    res = []
    for word in sentence:
        
        if word.lower() not in word_to_idx:
            res.append(word_to_idx["<unk>"])
        else:
            res.append(word_to_idx[word.lower()])
    return res

def idx_to_label(idx_to_labels:dict, src_label:List[List[int]]):
    """Converts list of labels indexes to their string value. It's the inverse operation of label_to_idx function

    Args:
        labels_to_idx (dict): dictionary with structure {label:index}
        src_label (List[List[int]]): list of label indexes

    Returns:
        List[List[str]]: List of list of labels (strings)
    """
    out_label = []
    temp = []
    for label_list in src_label:
        
        temp = []
        for label in label_list:
            
            if '<pad>' == idx_to_labels[int(label)]:
                temp.append("O") 
            else:
                temp.append(idx_to_labels[label])
                    
        

        out_label.append(temp)

                  
    return out_label


def plot_logs(logs, title): #notebook 3

    plt.figure(figsize=(8,6))

    plt.plot(list(range(len(logs['train_history']))), logs['train_history'], label='Train loss')
    plt.plot(list(range(len(logs['valid_history']))), logs['valid_history'], label='Valid loss')
    plt.plot(list(range(len(logs['f1_history']))), logs['f1_history'], label='F1',color = 'red')

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")

    plt.show()

def collate_fn(sentence):
    """Collate function for the dataloader for batch padding

    Args:
        sentence (list(list(str),list(str))): List of list of couples [[sentence],[labels]]

    Returns:
        Tensor: padded sentence
        Tensor: padded labels
        list(int): lenghts of  non padded sentence
        list(int): lenghts of  non padded labels
        
    """

    (sentences, labels) = zip(*sentence)

    tensor_sentences = [torch.tensor(sentence_) for sentence_ in sentences ]     
    tensor_labels = [torch.tensor(label) for label in labels ]

    sentences_lens = [len(sentence_) for sentence_ in tensor_sentences]
    labels_lens = [len(label) for label in tensor_labels]

    tensor_sentences_padded = pad_sequence(tensor_sentences, batch_first=True, padding_value=0)
    tensor_labels_padded = pad_sequence(tensor_labels, batch_first=True, padding_value=0)
    
    return {'sentence': tensor_sentences_padded.to(DEVICE),'labels': tensor_labels_padded.to(DEVICE), 'sentence_len': sentences_lens, 'labels_len': labels_lens}
    

def str_to_int(str_dict_key:dict):
    """Function for dict keys conversion from str to int

    Args:
        str_dict_key (dict): dictionary with keys having str type

    Returns:
        dict: input dict with the same keys converted in integers
    """
    dict = {}
    for key in str_dict_key:
        dict[int(key)] = str_dict_key[key]
    return dict


   