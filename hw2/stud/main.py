import utils
import os
import vocabulary
import wsddataset


training_data = utils.build_data_from_jsonl(
    os.path.join(utils.DIRECTORY_NAME, '../../data/coarse-grained/train_coarse_grained.json'))
valid_data = utils.build_data_from_jsonl(
    os.path.join(utils.DIRECTORY_NAME, '../../data/coarse-grained/val_coarse_grained.json'))
test_data = utils.build_data_from_jsonl(
    os.path.join(utils.DIRECTORY_NAME, '../../data/coarse-grained/test_coarse_grained.json'))

#print(training_data)
#As vocabulary init class accept lists as input, this aux function does the conversion
list_candidates = utils.list_all_values(training_data,'candidates')
vocab = vocabulary.Vocabulary(training_data["words"],list_candidates)
print(vocab.idx_to_labels)

train_dataset = wsddataset.WsdDataset(training_data["words"],training_data["senses"],training_data["instance_ids"],vocab.word_to_idx,vocab.labels_to_idx)
print(train_dataset[150])