import os
import json
import torch
from typing import List

from allennlp.modules.elmo import batch_to_ids



###HYPERPARAMETERS###

NUM_WORKERS = 12
EMBEDDING_DIM = 300
LAYERS_NUM = 2
HIDDEN_DIM = 150
EPOCHS_NUM = 500
LEARNING_RATE = [1e-2] #i
ELMO_LR = [1e-5] #j
LSTM_LR = [1e-3]
CHANCES = 5
DROPOUT_LAYER = [0.5,0.8] #k
DROPOUT_EMBED = [0.5]
DROPOUT_LSTM = 0.2
BATCH_SIZE = 600
LIN_WD = [0,0.001, 0.01]
ELMO_WD = [0,0.001, 0.01]
LSTM_WD = [0,0.001, 0.01]
#####################

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


def extract_labels_and_sense_indices(batch: List[dict]):
    """Extract labels (senses) and target word indices for all the sentences

    Args:
        batch (List[dict]): sample dict

    Returns:
        tuple: (labels , indices) where both values are tensors
    """
    labels = []
    idx = []
    for sentence in batch:

        label = sentence["sample"]["senses"]
        temp = []

        for index in label:

            temp.append(int(index))
            labels.append(label[index])

        idx.append(torch.tensor(temp)) #one sentence may have more words to disambiguate

    idx = torch.nn.utils.rnn.pad_sequence(
        idx, batch_first=True, padding_value=-1)
    
    labels = torch.as_tensor(labels)

    return labels, idx





def collate_fn_elmo(batch):
    
    batch_out = {}
    batch_out["input_ids"] = batch_to_ids([sentence["sample"]["words"] for sentence in batch])
    
    labels, idx = extract_labels_and_sense_indices(batch)
    
    batch_out["labels"] = labels
    batch_out["idx"] = idx
    
    return batch_out

def collate_fn_glove(batch):
    
    batch_out = {}
    batch_out["input_ids"] = [sentence["sample"]["words"] for sentence in batch]
    
    labels, idx = extract_labels_and_sense_indices(batch)
    
    batch_out["labels"] = labels
    batch_out["idx"] = idx
   
    return batch_out

def get_idx_from_tensor(tensor_idx)-> List[tuple]:
    """Aux function for get_senses_vector: recover the word indices of the words to disambiguate from tensor_idx.
    

    Args:
        tensor_idx (Tensor): Tensor containig target words indices for each sentence

    Returns:
        List[tuple]: one tuple for each sentence containing the indices for target words
    """
    idx = []
    for row in tensor_idx:
        temp = []
        for elem in row:
            if elem == -1: # -1 is the chosen pad value
                break
            temp.append(int(elem))
        idx.append(tuple(temp))
        

    return idx

def get_senses_vector(model_output, tensor_idx, word_ids):
    """This function extracts the vector embedding for each target words for each sentence

    Args:
        model_output (Tensor): transformer output tensor
        tensor_idx (Tensor): Tensor where each row contains all sentence original target word indices
        word_ids (Tensor): Tensor where each row contains all sentence new target word indices (after suubwords tokenizations)
    Returns:
        Tensor: The stacked tensor of all target words embedding
    """
    
    res = []
    for i in range(model_output.size(0)):
        for elem in tensor_idx[i]:
            if elem ==-1:
                break
            res.append(model_output[i][elem])
    

    res = torch.stack(res, dim=0)
    
    return res

def glove_embedding_tensorization(input_ids,embedding):
    stack = []
    for sentence in input_ids:
        stack.append(embedding.get_vecs_by_tokens(sentence))
    
    stack =  torch.nn.utils.rnn.pad_sequence(stack,batch_first=True,padding_value=-100)
    return stack.to("cuda")