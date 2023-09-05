import os
import xml.etree.ElementTree as ET
import time
dir = os.path.dirname(__file__)

import os
import xml.etree.ElementTree as ET

def parse_xml(xml_string, labels_f):
    root = ET.fromstring(xml_string)
    
    
    labels_line = labels_f.readline().strip().split()
    if labels_line != []:
                    
        labels_line = labels_line[1].split(":")[1][0:-1]
    sentences = []
    idx = 0
    for sentence_elem in root.findall('.//sentence'):
        
        sentence_parts = ""
        instances = {"words": {}, "senses": {}, "pos": {}}
        
        for part in sentence_elem:
            if part.tag == 'wf':
                sentence_parts += " " + part.text
                idx+=1
            elif part.tag == 'instance':

                
                instances["senses"][idx] = labels_line
                instances["pos"][idx] = part.attrib["pos"]
                sentence_parts += " " + part.text
                idx +=1
                labels_line = labels_f.readline().strip().split()
                if labels_line != []:
                    
                    labels_line = labels_line[1].split(":")[1][0:-1] #l'elemento uno ha struttura bn:1234n. Così facendo splitto sui ":" per prendere il numero e rimuovo la lettera finale
            instances["words"] = sentence_parts
        
        
        sentences.append(instances)
        idx = 0
    
    return sentences
def process_folder(root_folder):
    result = []
    
    for root, dirs, files in os.walk(root_folder):
        for folder in dirs:
            if folder != "semcor_it":
                continue
            xml_path = os.path.join(root, folder, f"{folder}.data.xml")
            labels_path = os.path.join(root, folder, f"{folder}.gold.key.txt")
            if os.path.exists(xml_path) and os.path.exists(labels_path):
                with open(xml_path, 'r') as xml_f, open(labels_path, 'r') as labels_f:
                    xml_content = xml_f.read()
                    
                    
                    parsed_data = parse_xml(xml_content, labels_f)
                    
                    result.extend(parsed_data)
                    

    return result


root_folder = os.path.join(dir,"xl-wsd/training_datasets/")
parsed_results = process_folder(root_folder)

for sentence_data in parsed_results:
    print("Frase completa:", sentence_data['words'])
    print("Dizionario delle istanze:", sentence_data['senses'])
    print("POS:" , sentence_data['pos'])

    print()
