import json
import os
from typing import List
from transformers import AutoTokenizer
import torch
import time
import numpy as np
import evaluate
import copy
# we will use with Distil-BERT
#language_model_name = "distilbert-base-uncased"
# this GPU should be enough for this task to handle 32 samples per batch
BATCH = 128
# we keep num_workers = min(4 * number of GPUs, number of cores)
# tells the data loader how many sub-processes to use for data loading
num_workers = 12
# optim
learning_rate = 1e-3
weight_decay = 0.0
transformer_learning_rate = 1e-5
transformer_weight_decay = 0.0
# training
epochs = 3
device = "cuda" if torch.cuda.is_available() else "cpu"
LANGUAGE_MODEL_NAME = "bert-base-cased"
DIRECTORY_NAME = os.path.dirname(__file__)
tokenizer =  AutoTokenizer.from_pretrained("bert-base-cased", use_fast = False)
#tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
seqeval = evaluate.load("f1")

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
    #line = f.readline()
    words = []
    candidates = [] # labels
    lemmas = []
    instance_ids = []
    pos_tags = []
    senses = []
    samples = []
    data = json.load(f)
    for json_line in data:
        samples.append({"instance_ids": data[json_line]["instance_ids"],"lemmas": data[json_line]["lemmas"], "words": data[json_line]["words"],"pos_tags": data[json_line]["pos_tags"],"senses":data[json_line]["senses"],"candidates":data[json_line]["candidates"]})
        words.append(data[json_line]["words"])
        #print(data[json_line]["words"])
        #time.sleep(5)
        #candidates.append(data[json_line]["candidates"])

        #lemmas.append(data[json_line]["lemmas"])
        #pos_tags.append(data[json_line]["pos_tags"])
        #senses.append(data[json_line]["senses"])
        
        #instance_ids.append(data[json_line]["instance_ids"])
        
    
    f.close()

    return {
        "samples": samples,
        "words": words
        #'instance_ids': instance_ids,
        #'lemmas': lemmas,
        #'pos_tags': pos_tags,
        #'senses': senses,
        #'words': words,
        #'candidates': candidates

    }

def list_all_values(data:List[dict],key:str):
    """Take a list of dictionaries and return al list of all the values 

    Args:
        data (List[dict]): list of dict 
        key (str): key of the dict to process

    Returns:
        List[str]: List containing all the values
    """
    #print(data)
    time.sleep(5)
    if key not in data:
        raise "NOT VALID INPUT KEY, KEY NOT FOUND IN DICTIONARY"
    labels = []
    for candidate in data[key]:
        for key in candidate:
            labels.append(candidate[key])
    return labels

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

def label_to_idx(labels_to_idx:dict, labels: List[str]):
    #print(labels)
    """Converts labels string in integer indexes. 
       

    Args:
        labels_to_idx (dictionary): dictionary with structure {label:index} 
        labels (List[string]): List of labels (stings)

    Returns:
        list: list of integers that represent labels indexes
    """
    
    labels = copy.deepcopy(labels)
    
    for label in labels:
        

        temp = []
        #print(labels[label])
        labels[label] = labels_to_idx[labels[label][0]]
            #temp.append(labels_to_idx[value])
        #print(labels)
       #labels[label] = temp
        #print(labels)
        #time.sleep(5)

    return labels


def build_all_senses(file_path):
    try:
        f = open(file_path, 'r')
    except OSError:
        print("Unable to open file in "+ str(file_path))
    #line = f.readline()
    senses = []
    data = json.load(f)
    for json_line in data:
        #senses.append({json_line:data[json_line]}) uncomment for fine grained operations
        senses.append([json_line])

        


    f.close()
    #print(senses)
    return senses

def collate_fn(batch):
    #print("COLLATOOOOO")
    
    batch_out = tokenizer(
        [sentence["sample"]["words"] for sentence in batch],
        return_tensors="pt",
        padding=True,
        # We use this argument because the texts in our dataset are lists of words.
        is_split_into_words=True,
    )
    
    
    labels,idx = extract_labels_and_senses_index(batch)   
    batch_out["labels"] = labels
    batch_out["idx"] = idx
    
    return batch_out

def compute_metrics(labels,outputs,label_list):
    outputs = outputs.argmax(dim=1)
    #print(outputs)
    y_true = labels.tolist()
    y_pred = outputs.tolist()
    predictions, labels = [], []
    #print(y_pred)
    true_predictions = [
        [label_list[p] for (p, l) in zip(y_pred, y_true) if l != -100 ]
        
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(y_pred, y_true) if l != -100]
        
    ]
    predictions += true_predictions
    labels += true_labels



    
    return seqeval.compute(predictions=y_pred, references=y_true, average = 'macro')
         
     


def extract_labels_and_senses_index(batch):
    


    
    labels = []
    idx = []
    #print(labels)
    for sentence in batch:
        #print(sentence)
        label = sentence["senses"]
        #print("Senses: " + str(label))
        temp = []
        for index in label:
            #print("index: " + str(index))
            temp.append(int(index))
            labels.append(label[index])
            #print("List: " + str(labels))
        idx.append(torch.tensor(temp))
        #time.sleep(5)

    idx = torch.nn.utils.rnn.pad_sequence(idx, batch_first=True, padding_value=-1)
    #print("List: " + str(labels))
    #print(idx)
    labels = torch.as_tensor(labels)

    return labels , idx

def get_idx_from_tensor(tensor_idx):
    idx = []
    for row in tensor_idx:
        #print(row)
        temp = []
        for elem in row:
            if elem == -1:
                break
            temp.append(int(elem))
        idx.append(tuple(temp))
        #print(idx)
        #time.sleep(2)
    
    return idx
def get_senses_vector(model_output, tensor_idx):
    idx = get_idx_from_tensor(tensor_idx)
    res = []
    for i in range(model_output.size(0)):
        
        for elem in range(len(idx[i])):
        
            y = torch.stack((model_output[i][0],model_output[i][idx[i][elem]]), dim = -2)
            sum = torch.sum(y, dim = -2)
            res.append(sum)

    res = torch.stack(res, dim = -2)
    return res