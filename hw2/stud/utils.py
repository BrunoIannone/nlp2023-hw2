import json
import os
from typing import List
from transformers import AutoTokenizer
import torch
import time
import evaluate
# we will use with Distil-BERT
#language_model_name = "distilbert-base-uncased"
# this GPU should be enough for this task to handle 32 samples per batch
batch_size = 128
# we keep num_workers = min(4 * number of GPUs, number of cores)
# tells the data loader how many sub-processes to use for data loading
num_workers = 1
# optim
learning_rate = 1e-3
weight_decay = 0.0
transformer_learning_rate = 1e-5
transformer_weight_decay = 0.0
# training
epochs = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
LANGUAGE_MODEL_NAME = "bert-base-cased"
DIRECTORY_NAME = os.path.dirname(__file__)
tokenizer =  AutoTokenizer.from_pretrained("bert-base-cased")
#tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
seqeval = evaluate.load("seqeval")

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
    data = json.load(f)
    for json_line in data:
        for sense in data[json_line]["senses"]:
            words.append(data[json_line]["words"])
            candidates.append(data[json_line]["candidates"][sense])

            lemmas.append(data[json_line]["lemmas"])
            pos_tags.append(data[json_line]["pos_tags"])
            senses.append(data[json_line]["senses"][sense])
            instance_ids.append(data[json_line]["instance_ids"][sense])
        
    

    f.close()

    return {

        'instance_ids': instance_ids,
        'lemmas': lemmas,
        'pos_tags': pos_tags,
        'senses': senses,
        'words': words,
        'candidates': candidates,

    }

def list_all_values(data:List[dict],key:str):
    """Take a list of dictionaries and return al list of all the values 

    Args:
        data (List[dict]): list of dict 
        key (str): key of the dict to process

    Returns:
        List[str]: List containing all the values
    """
    print(data)
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
    res = []

    for label in labels:
        if label is None:
            res.append(-100) 
        else:
            res.append(labels_to_idx[label])
    return res


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
    batch_out = tokenizer(
        [sentence["tokens"] for sentence in batch],
        return_tensors="pt",
        padding=True,
        # We use this argument because the texts in our dataset are lists of words.
        is_split_into_words=True,
    )
    
    labels = [sentence["ner_tags"] for sentence in batch]
    
    batch_out["labels"] = torch.as_tensor(labels)
    return batch_out

def compute_metrics(labels,outputs,label_list):
    outputs = outputs.argmax(dim=-1)
    #print(outputs)
    y_true = labels.tolist()
    y_pred = outputs.tolist()
    predictions, labels = [], []

    true_predictions = [
        [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100 ]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    predictions += true_predictions
    labels += true_labels



    results = seqeval.compute(predictions=predictions, references=labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }