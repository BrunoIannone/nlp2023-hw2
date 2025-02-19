import os
import torch
from typing import List
from allennlp.modules.elmo import batch_to_ids



### GENERAL HYPERPARAMETERS ###

NUM_WORKERS = 12
EPOCHS_NUM = 500
DROPOUT_LAYER = [0.5] #k
DROPOUT_EMBED = [0.5]
LEARNING_RATE = [1e-2] #i
LIN_WD = [0, 0.001, 0.01, 0.1]

### ELMO HYPERPARAMETERS ###
ELMO_BATCH_SIZE = 32
ELMO_LR = [1e-5] #j
ELMO_HIDDEN_DIM = 512
ELMO_WD = [0, 0.001, 0.01]

### GLOVE HYPERPARAMETERS ###
GLOVE_BATCH_SIZE = 600
EMBEDDING_DIM = 300
HIDDEN_DIM = 150
BIDIRECTIONAL = True
LAYERS_NUM = 2
DROPOUT_LSTM = 0.2
LSTM_LR = [1e-3]
LSTM_WD = [0, 0.001, 0.01, 0.1]

############################


DIRECTORY_NAME = os.path.dirname(__file__)

def extract_labels_and_sense_indices(batch: List[dict]):
    """Extract labels (senses) and target word indices for all the sentences

    Args:
        batch (List[dict]): List of sample dicts

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

def collate_fn_elmo(batch:List[dict]):
    """Collate function for ELMo WSD
        Args:
        batch (List[dict]): List of samples' dict
    Returns:
        dict: {"input_ids" : Tensor, "labels": Tensor, "idx": Tensor } N.B. idx is the tensor containing target word indices padded with -1
    """
    
    batch_out = {}
    batch_out["input_ids"] = batch_to_ids([sentence["sample"]["words"] for sentence in batch])
    
    labels, idx = extract_labels_and_sense_indices(batch)
    
    batch_out["labels"] = labels
    batch_out["idx"] = idx
    
    return batch_out




def get_target_vector(model_output: torch.Tensor, tensor_idx:torch.Tensor):
    
    """
    This function extracts the vector embedding for each target words for each sentence

    Args:
        model_output (Tensor): transformer output tensor
        tensor_idx (Tensor): Tensor where each row contains all sentence original target word indices
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




def collate_fn_glove(batch: List[dict]):
    """Collate function for GloVe WSD
    Args:
        batch (List[dict]): List of samples' dict

    Returns:
        dict: {"input_ids" : List[List[str]], "labels": Tensor, "idx": Tensor } N.B. idx is the tensor containing target word indices padded with -1
    """
    
    batch_out = {}
    batch_out["input_ids"] = [sentence["sample"]["words"] for sentence in batch]
    
    labels, idx = extract_labels_and_sense_indices(batch)
    
    batch_out["labels"] = labels
    batch_out["idx"] = idx
   
    return batch_out


def glove_embedding_tensorization(input_ids: List[List[str]],embedding):
    """Build the tensor of word embeddings for each sentence in the batch

    Args:
        input_ids (List[List[str]]): List of List of sentence tokens
        embedding: GloVe instance

    Returns:
        Tensor: Tensor containing the GloVe word embedding for each sentence (batch_size x max_length x embedding_dim )
    """
    stack = []
    for sentence in input_ids:
        stack.append(embedding.get_vecs_by_tokens(sentence))
    
    stack =  torch.nn.utils.rnn.pad_sequence(stack,batch_first=True,padding_value=-100)
    return stack.to("cuda")