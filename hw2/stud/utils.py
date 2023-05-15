import json
import os
from typing import List



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
    #line = f.readline()
    words = []
    candidates = [] # labels
    lemmas = []
    instance_ids = []
    pos_tags = []
    senses = []
    data = json.load(f)
    for json_line in data:
        words.append(data[json_line]["words"])
        candidates.append(data[json_line]["candidates"])
        lemmas.append(data[json_line]["lemmas"])
        pos_tags.append(data[json_line]["pos_tags"])
        senses.append(data[json_line]["senses"])
        instance_ids.append(data[json_line]["instance_ids"])
        


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
    if key not in data:
        raise "NOT VALID INPUT KEY, KEY NOT FOUND IN DICTIONARY"
    labels = []
    for candidate in data[key]:
        for key in candidate:
            labels.append(candidate[key])
    return labels