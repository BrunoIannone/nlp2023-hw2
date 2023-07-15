import json
import os
from typing import List
from transformers import AutoTokenizer
import torch
import time

BATCH_SIZE = 2
NUM_WORKERS = 12
LEARNING_RATE = 1e-3
weight_decay = 0.0
transformer_learning_rate = 1e-5
transformer_weight_decay = 0.0
NUM_EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LANGUAGE_MODEL_NAME = "distilbert-base-uncased"
# LANGUAGE_MODEL_NAME = "roberta-base"
LANGUAGE_MODEL_NAME = 'kanishka/GlossBERT'
LANGUAGE_MODEL_NAME_POS =  'QCRI/bert-base-multilingual-cased-pos-english'

DIRECTORY_NAME = os.path.dirname(__file__)
LANGUAGE_MODEL_PATH = os.path.join(DIRECTORY_NAME, '../../model/GlossBERT')
TOKENIZER = AutoTokenizer.from_pretrained(
    LANGUAGE_MODEL_NAME, use_fast=True, add_prefix_space=True)


def build_data_from_json(file_path: str, save_words: bool = False):
    """Retrieve samples and sample sentences from JSON file (if save_words = True)

    Args:
        file_path (string): path to JSON file
        save_words (bool) : save samples' sentences
    Returns:
        if (save_words = True)
            dictionary: {
                samples: List[{"instance_ids": {"id" (int): str}, "lemmas": List[str], "words": List[str], "pos_tags" = List[str], "senses": {"word_index" (int): List[str] (senses)},"candidates": {"word_index" (int): List[str](candidate senses)}}], 
                words: List[List[str]]
            } 
        else:
            dictionary: {
                samples: List[{"instance_ids": {"id" (int): str}, "lemmas": List[str], "words": List[str], "pos_tags" = List[str], "senses": {"word_index" (int): List[str] (senses)},"candidates": {"word_index" (int): List[str](candidate senses)}}], 

            }

    """
    try:
        f = open(file_path, 'r')
    except OSError:
        print("Unable to open file in " + str(file_path))
    if (save_words):
        words = []
    samples = []
    data = json.load(f)
    for json_line in data:
        samples.append({"instance_ids": data[json_line]["instance_ids"], "lemmas": data[json_line]["lemmas"], "words": data[json_line]["words"],
                       "pos_tags": data[json_line]["pos_tags"], "senses": data[json_line]["senses"], "candidates": data[json_line]["candidates"]})
        if (save_words):
            words.append(data[json_line]["words"])

    f.close()

    if (not save_words):
        return {
            "samples": samples,

        }
    else:
        return {
            "samples": samples,
            "words": words,

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


def label_to_idx(labels_to_idx: dict, labels_idx_dict: List[str]):
    # print(labels)
    """Converts labels string in integer indexes. 


    Args:
        labels_to_idx (dictionary): dictionary with structure {label:index} 
        labels_idx_dict (dict): {"word_index": List[str](senses)}

    Returns:
        dict: {"word_index": List[int](senses)}, a word could be associated with more senses, the first one is taken in account
        """

    res = {}
    #print("Labels_dict")
    #print(labels_idx_dict)
    #time.sleep(2)
    modificato = False
    for word_idx in labels_idx_dict:
        #print(word_idx)
        #time.sleep(10)
        if labels_idx_dict[word_idx][0] not in list(labels_to_idx.keys()) and len(labels_idx_dict[word_idx])<=1 :
            #print("NO")
            #print(labels_idx_dict)
            #time.sleep(10)
            res[word_idx] = labels_to_idx['O']
        elif len(labels_idx_dict[word_idx])>1: ##in un futuro qua vanno usati i gloss
            for sense in labels_idx_dict[word_idx]:
                #print("SENSE" +str(sense))
                if sense in labels_to_idx:
                    #print("SI>1")
                    #print(labels_idx_dict[word_idx])
                    #print(sense)
                    res[word_idx] = labels_to_idx[sense]
                    #print(res)
                    #time.sleep(10)
                    modificato = True
                    break
            if not modificato:
                res[word_idx] = labels_to_idx['O']
            

                    
        
        else:
            #print("GIA")
            res[word_idx] = labels_to_idx[labels_idx_dict[word_idx][0]]
    #print("RES" + str(res))
    #time.sleep(20)
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
    return senses


def collate_fn(batch: List[dict]):
    """collate_fn for DataLoader

    Args:
        batch (List[dict]): List of samples' 

    Returns:
        dict: { "word_ids" = word_ids,"labels" = labels,"idx" = idx} all values are tensors
    """
    batch_out = TOKENIZER(
        [sentence["sample"]["words"] for sentence in batch],
        return_tensors="pt",
        padding=True,
        is_split_into_words=True,
        truncation=True

    )
    word_ids = map_new_index(batch, batch_out) #needed to take in account the words shift after tokenization
    labels, idx = extract_labels_and_sense_indices(batch) #extract word indices and relative label

    batch_out["word_ids"] = word_ids
    batch_out["labels"] = labels
    batch_out["idx"] = idx

    return batch_out


def map_new_index(batch:List[dict], batch_out:dict):
    """Aux function for callate_fn: recovers word_ids from tokenized sentences

    Args:
        batch (List[dict]): List of samples' 
        batch_out (dict): batch after tokenization

    Returns:
        Tensor: word_ids for each sample words
    """
    word_ids = []
    for idx, sentence in enumerate(batch):
        word_ids.append(batch_out.word_ids(batch_index=idx))
    last_index = None
    res = []
    i = 0
    for l in word_ids:
        i = 0
        temp = []
        last_index = None
        while (i < len(l)):
            if last_index != None and l[last_index] == l[i]:
                i += 1

                continue
            else:
                temp.append(i)
                last_index = i
                i += 1
        res.append(torch.tensor(temp))

    word_ids_ = torch.nn.utils.rnn.pad_sequence(
        res, batch_first=True, padding_value=-1) # -1 is the chosen pad value
    return word_ids_


def extract_labels_and_sense_indices(batch: List[dict]) -> tuple: # tuple of tensors(labels,indices)
    """Extract labels (senses) and target word indices for all the sentences

    Args:
        batch ( List[dict]): sample dict

    Returns:
        tuple: (labels , indices) both values are tensors
    """
    labels = []
    idx = []
    # print(labels)
    for sentence in batch:
        # print(sentence)
        label = sentence["sample"]["senses"]
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


def get_idx_from_tensor(tensor_idx)-> List[tuple]:
    """Aux function for get_senses_vector: recover the word indices of the words to disambiguate from tensor_idx.
    

    Args:
        tensor_idx (Tensor): Tensor containig target words indices for each sentence

    Returns:
        List[tuple]: one tuple for each sentence containing the indices for target words
    """
    idx = []
    for row in tensor_idx:
        # print(row)
        temp = []
        for elem in row:
            if elem == -1: # -1 is the chosen pad value
                break
            temp.append(int(elem))
        idx.append(tuple(temp))
        # print(idx)
        # time.sleep(2)

    return idx


def get_senses_vector(model_output, tensor_idx, word_ids):
    """This function extracts the vector embedding for each target words for each sentence

    Args:
        model_output (Tensor): transformer output tensor
        tensor_idx (Tensor): Tensor containing all sentence target word indices

    Returns:
        Tensor: The stacked tensor of all target words embedding
    """
    idx = get_idx_from_tensor(tensor_idx)
    res = []
    # print(model_output.size())

    for i in range(model_output.size(0)):
        # print(idx[i])

        for elem in range(len(idx[i])):
            # y = torch.stack(
            #    (model_output[i][0], model_output[i][idx[i][elem]]), dim=-2)
            # sum = torch.sum(y, dim=-2)
            # res.append(sum)
            # print(model_output[i][idx[i][elem]])

            # time.sleep(5)
            # res.append(model_output[i][0])
            original_index = idx[i][elem]
            # print("orig:" + str(original_index))
            shifted_index = int(word_ids[i][original_index])
            # print("Shift: " + str(shifted_index))
            lenght = int(word_ids[i][original_index + 1]) - shifted_index
            # print(lenght)
            # time.sleep(5)
            stack = torch.stack(
                [model_output[i][shifted_index: shifted_index + lenght]], dim=1)
            # print(stack.size())
            # time.sleep(2)
            res.append(torch.sum(stack, dim=0).squeeze())
            # print(res)

    res = torch.stack(res, dim=-2)
    return res


def str_to_int(str_dict_key: dict) ->dict :
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


def idx_to_label(idx_to_labels: dict, src_label: List[List[int]]) -> List[List[str]]:
    """Converts list of labels indexes to their string value. It's the inverse operation of label_to_idx function

    Args:
        labels_to_idx (dict): dictionary with structure {label:index}
        src_label (List[List[int]]): list of label indexes

    Returns:
        List[List[str]]: List of list of labels (strings)
    """
    # print(src_label)
    out_label = []
    temp = []
    # for label_list in src_label:

    # temp = []
    for label in src_label:

        if '<pad>' == idx_to_labels[int(label)]:
            out_label.append("O")
        else:
            out_label.append(idx_to_labels[label])

    # out_label.append(temp)

    return out_label
