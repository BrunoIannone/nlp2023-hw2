import json
import os
from typing import List
from transformers import AutoTokenizer
import torch
import time
import copy
BATCH_SIZE = 8
NUM_WORKERS = 12
LEARNING_RATE = 1e-3
weight_decay = 0.0
transformer_learning_rate = 1e-5
transformer_weight_decay = 0.0
NUM_EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#LANGUAGE_MODEL_NAME = "distilbert-base-uncased"
#LANGUAGE_MODEL_NAME = "roberta-base"

LANGUAGE_MODEL_NAME = 'kanishka/GlossBERT'
DIRECTORY_NAME = os.path.dirname(__file__)
TOKENIZER = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME, use_fast=True,add_prefix_space = True)


def build_data_from_json(file_path: str):
    """Retrieve samples and sample sentences from JSON file

    Args:
        file_path (string): path to JSON file

    Returns:
        dictionary: {
            samples: List[{"instance_ids": {"id" (int): str}, "lemmas": List[str], "words": List[str], "pos_tags" = List[str], "senses": {"word_index" (int): List[str] (senses)},"candidates": {"word_index" (int): List[str](candidate senses)}}], 
            words: List[List[str]]
        }
    """
    try:
        f = open(file_path, 'r')
    except OSError:
        print("Unable to open file in " + str(file_path))
    # line = f.readline()
    words = []
    samples = []
    labels = []
    data = json.load(f)
    for json_line in data:
        samples.append({"instance_ids": data[json_line]["instance_ids"], "lemmas": data[json_line]["lemmas"], "words": data[json_line]["words"],
                       "pos_tags": data[json_line]["pos_tags"], "senses": data[json_line]["senses"], "candidates": data[json_line]["candidates"]})
        words.append(data[json_line]["words"])
        #temp = [data[json_line]["candidates"][sense] for sense in data[json_line]["candidates"]]
        #labels.extend(temp)
    f.close()

    return {
        "samples": samples,
        "words": words,
        #"labels":labels
    }


def word_to_idx(word_to_idx: dict, sentence: List[str]):
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


def label_to_idx(labels_to_idx: dict, labels: List[str]):
    # print(labels)
    """Converts labels string in integer indexes. 


    Args:
        labels_to_idx (dictionary): dictionary with structure {label:index} 
        labels (dict): {"word_index": List[str](senses)}

    Returns:
        dict: {"word_index": List[int](senses)}, a word could be associated with more senses, the first one is taken in account
        """

    res = {}
    for label in labels:

        res[label] = labels_to_idx[labels[label][0]]
        
    return res


def build_all_senses(file_path):
    """Get all senses

    Args:
        file_path (str): file containing all coarse_grained senses

    Returns:
        List[List[str]]: List containing list of senses
    """
    try:
        f = open(file_path, 'r')
    except OSError:
        print("Unable to open file in " + str(file_path))
    # line = f.readline()
    senses = []
    data = json.load(f)
    for json_line in data:
        # senses.append({json_line:data[json_line]}) uncomment for fine grained operations
        # converting to list because of vocabulary function input type
        senses.append([json_line])

    f.close()
    # print(senses)
    return senses


def collate_fn(batch,prediction):
    
    #print(batch)
    #time.sleep(5)
    batch_out = TOKENIZER(
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
    #print(word_ids)
    #time.sleep(5)
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
    labels, idx = extract_labels_and_sense_indices(batch)

    batch_out["word_ids"] = word_ids_
    #print( batch_out["word_ids"])
    #time.sleep(5)
    batch_out["labels"] = labels
    batch_out["idx"] = idx

    return batch_out


def extract_labels_and_sense_indices(batch):
    """Extract labels (senses) and target word indices for all the sentenses

    Args:
        batch (dict): sample dict

    Returns:
        tuple: (labels , indices) both values are tensors
    """
    labels = []
    idx = []
    # print(labels)
    for sentence in batch:
        # print(sentence)
        label = sentence["senses"]
        # print("Senses: " + str(label))
        temp = []
        for index in label:
            # print("index: " + str(index))
            temp.append(int(index))
            labels.append(label[index])
            # print("List: " + str(labels))
        idx.append(torch.tensor(temp))
        # time.sleep(5)

    idx = torch.nn.utils.rnn.pad_sequence(
        idx, batch_first=True, padding_value=-1)
    # print("List: " + str(labels))
    # print(idx)
    labels = torch.as_tensor(labels)

    return labels, idx


def get_idx_from_tensor(tensor_idx):
    """Recover the word indices of the words to disambiguate from tensor_idx.
    Is an auxiliary function for get_senses_vector

    Args:
        tensor_idx (Tensor): Tensor containig target words for each sentence

    Returns:
        List[tuple]: one tuple for each sentence containing the indices for target words
    """
    idx = []
    for row in tensor_idx:
        # print(row)
        temp = []
        for elem in row:
            if elem == -1:
                break
            temp.append(int(elem))
        idx.append(tuple(temp))
        # print(idx)
        # time.sleep(2)

    return idx


def get_senses_vector(model_output, tensor_idx, word_ids):
    """This function extracts the vector embedding for each target word for each sentence

    Args:
        model_output (Tensor): transformer output tensor
        tensor_idx (Tensor): Tensor containing all sentence target word indices

    Returns:
        Tensor: The stacked tensor of all target words embedding
    """
    idx = get_idx_from_tensor(tensor_idx)
    res = []
    #print(model_output.size())
    
    for i in range(model_output.size(0)):
        #print(idx[i])

        for elem in range(len(idx[i])):
            #y = torch.stack(
            #    (model_output[i][0], model_output[i][idx[i][elem]]), dim=-2)
            #sum = torch.sum(y, dim=-2)
            #res.append(sum)
            #print(model_output[i][idx[i][elem]])

            #time.sleep(5)
            #res.append(model_output[i][0])
            original_index = idx[i][elem]
            #print("orig:" + str(original_index))
            shifted_index = int(word_ids[i][original_index])
            #print("Shift: " + str(shifted_index))
            lenght = int(word_ids[i][original_index + 1]) - int(word_ids[i][original_index]) 
            #print(lenght)
            #time.sleep(5)
            stack = torch.stack([model_output[i][shifted_index : shifted_index + lenght]],dim = 1)
            #print(stack.size())
            #time.sleep(2)
            res.append(torch.sum(stack,dim = 0).squeeze())
            #print(res)
            

    res = torch.stack(res, dim=-2)
    return res

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


def idx_to_label(idx_to_labels:dict, src_label:List[List[int]]):
    """Converts list of labels indexes to their string value. It's the inverse operation of label_to_idx function

    Args:
        labels_to_idx (dict): dictionary with structure {label:index}
        src_label (List[List[int]]): list of label indexes

    Returns:
        List[List[str]]: List of list of labels (strings)
    """
    print(src_label)
    out_label = []
    temp = []
    #for label_list in src_label:
        
        #temp = []
    for label in src_label:
            
        if '<pad>' == idx_to_labels[int(label)]:
             out_label.append("O") 
        else:
           out_label.append(idx_to_labels[label])
                    
        

    #out_label.append(temp)

                  
    return out_label