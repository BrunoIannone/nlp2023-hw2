import json
import os
from typing import List
from transformers import AutoTokenizer
import torch
import time

BATCH_SIZE = 16
NUM_WORKERS = 12
LEARNING_RATE = [1e-3]
weight_decay = [0,0.001,0.1]
transformer_learning_rate = [1e-5]
transformer_weight_decay = [0,0.001,0.1]
LIN_DROPOUT = [0.2,0.5,0.8]
NUM_EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#LANGUAGE_MODEL_NAME = 'distilroberta-base'
#LANGUAGE_MODEL_NAME = "distilbert-base-uncased"
#LANGUAGE_MODEL_NAME = "roberta-base"
LANGUAGE_MODEL_NAME = 'kanishka/GlossBERT'
#LANGUAGE_MODEL_NAME = 'bert-base-uncased'
#LANGUAGE_MODEL_NAME = "prajjwal1/bert-mini"
#LANGUAGE_MODEL_NAME_POS =  'QCRI/bert-base-multilingual-cased-pos-english'
#LANGUAGE_MODEL_NAME = "bert-base-cased"

DIRECTORY_NAME = os.path.dirname(__file__)

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
        #samples.append({"instance_ids": data[json_line]["instance_ids"], "lemmas": data[json_line]["lemmas"], "words": data[json_line]["words"],
         #               "pos_tags": data[json_line]["pos_tags"], "senses": data[json_line]["senses"], "candidates": data[json_line]["candidates"]})
        samples.append({"words": data[json_line]["words"],"senses": data[json_line]["senses"], "candidates": data[json_line]["candidates"]})
        
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
        sentence (List[str]): list of tokens (strings)

    Returns:
        List[int]: list of integers that represent tokens indexes
    """

    res = []
    for word in sentence:

        if word.lower() not in word_to_idx:
            res.append(word_to_idx["<unk>"])
        else:
            res.append(word_to_idx[word.lower()])
    return res


def label_to_idx(labels_to_idx: dict, target_word_idx:dict):
    """Converts labels string in integer indexes.
    If there is only one possible sense and it's in labels_to_idx, then its index is taken. If there are more possible senses, the first known sense found in labels_to_idx is taken.
    If there is only one sense and it's unknown or there are multiple senses but they are not known, the label corresponding to 'O' is taken.

    Args:
        labels_to_idx (dict): dictionary with structure {label:index} 
        target_word_idx (dict): {"target_word_index": List[str](senses)}

    Returns:
        dict: {"word_index": List[int](senses)}
        """

    res = {}
    modified = False

    for word_idx in target_word_idx:
        
        if target_word_idx[word_idx][0] not in list(labels_to_idx.keys()) and len(target_word_idx[word_idx])==1 : #Only one possible sense not in labels_to_idx
            
            res[word_idx] = labels_to_idx['O']
        
        elif len(target_word_idx[word_idx])>1: #More possible senses, take the first known
            ## //TODO: in un futuro qua vanno usati i gloss #
            
            for sense in target_word_idx[word_idx]:
               
                if sense in labels_to_idx:
                   
                    res[word_idx] = labels_to_idx[sense]
                    
                    modified = True
                    break
            if not modified:
                res[word_idx] = labels_to_idx['O']
        
        else:
            
            res[word_idx] = labels_to_idx[target_word_idx[word_idx][0]]
    
    return res


def build_all_senses(file_path:str, fine_grained:bool = False):
    """Get all senses from the mapping file

    Args:
        file_path (str): file containing coarse-grained mapping
        fine_grained (bool): True for fine grained operations. Default = False
    Returns:
        List[List[str]]: List of list of senses
    """
    try:
        f = open(file_path, 'r')
    except OSError:
        print("Unable to open file in " + str(file_path))

    senses = []
    data = json.load(f)
    
    for json_line in data:
        if fine_grained:
            for sense in data[json_line]:
                for key in sense:
                    senses.append([key])# converting to list because of vocabulary function input type

                    #print(key)
                    
        else:
            senses.append([json_line])# converting to list because of vocabulary function input type


    f.close()
    return senses


def collate_fn(batch: List[dict]):
    """collate_fn for DataLoader

    Args:
        batch (List[dict]): List of samples' dict

    Returns:
        dict: { "word_ids": word_ids,"labels": labels,"idx": idx} where all values are tensors
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
    """Aux function for collate_fn: recovers the new word indices after the tokenization.

    For example, if the sentence is "Luckly transformers are good", the tokenizer could divide the sentence in [[CLS],"Luck", "##ly","trans","##formers","are","good",[SEP]]; so this function will return Tensor([0,1,3,5,6,7]). 
    Keeping in account that the [CLS] token shifts everything by one, the word "transformers" which was before at index 1 is now at index 2. The value at index 2 is 3 which means that the first word has been splitted into two. With the same logic if we are looking for the word "are", we have to look for its original index plus one (so 2 + 1 = 3) where its value is 5 because of the tokenization of "transformers" and "Luckly".
    
    Args:
        batch (List[dict]): List of samples
        batch_out (dict): batch after word tokenization

    Returns:
        Tensor: new sentence word indices after tokenization. 
    
    """
    word_ids = []
    for idx, sentence in enumerate(batch):
        word_ids.append(batch_out.word_ids(batch_index=idx))

    last_index = None
    res = []
    i = 0
    for l in word_ids: #the code below finds in how many subwords a word has been divided as they will have the same index.
                       
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


def get_idx_from_tensor(tensor_idx):
    """Aux function for get_senses_vector: recovers the word indices of the words to disambiguate from tensor_idx.
    

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
        tensor_idx (Tensor): Tensor where each row contains all sentence original target word indices
        word_ids (Tensor): Tensor where each row contains all sentence new target word indices (after subwords tokenizations)
    Returns:
        Tensor: The stacked tensor of all target words embedding
    """
    #idx = get_idx_from_tensor(tensor_idx)
    res = []
    test_res = []
    # print(model_output.size())

    for i in range(model_output.size(0)):
        # print(idx[i])
        for elem in tensor_idx[i]:
            if elem ==-1:
                break
            #res.append(model_output[i][0])
            
           
            original_index = elem
            
            shifted_index = int(word_ids[i][original_index+1]) #the +1 takes in account the shift given by [CLS] token
            next_word = int(word_ids[i][original_index + 2]) #+1 from [CLS] +1 for next word
            word_lenght = next_word - shifted_index
            test_res.append((torch.sum(model_output[i][shifted_index: shifted_index + word_lenght], dim=0)/word_lenght))
          
        
        
    #     for elem in range(len(idx[i])):
            
    #         #res.append(model_output[i][0])
            
           
    #         original_index = idx[i][elem]
            
    #         shifted_index = int(word_ids[i][original_index+1]) #the +1 takes in account the shift given by [CLS] token
    #         next_word = int(word_ids[i][original_index + 2]) #+1 from [CLS] +1 for next word
    #         word_lenght = next_word - shifted_index
    #         res.append((torch.sum(model_output[i][shifted_index: shifted_index + word_lenght], dim=0)/word_lenght))
            
    # res = torch.stack(res, dim=0)
    test_res = torch.stack(test_res, dim=0)
    # if(not torch.eq(res,test_res)):
    #     print("DIVERSI DIOC")
    #     time.sleep(5)
    # else:
    #     print("AHA!!!!!!")
    return test_res


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


def idx_to_label(idx_to_labels: dict, src_label: List[List[int]]):
    """Converts list of labels indexes to their string value. It's the inverse operation of label_to_idx function

    Args:
        labels_to_idx (dict): dictionary with structure {label:index}
        src_label (List[List[int]]): list of label indexes

    Returns:
        List[List[str]]: List of list of labels (strings)
    """
    out_label = []
    
    
    for label in src_label:

        if '<pad>' == idx_to_labels[int(label)]:
            out_label.append("O")
        else:
            out_label.append(idx_to_labels[label])


    return out_label

