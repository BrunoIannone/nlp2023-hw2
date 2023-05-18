import utils
import os
import vocabulary
import wsddataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import train
import model as mod

training_data = utils.build_data_from_jsonl(
    os.path.join(utils.DIRECTORY_NAME, '../../data/coarse-grained/train_coarse_grained.json'))
valid_data = utils.build_data_from_jsonl(
    os.path.join(utils.DIRECTORY_NAME, '../../data/coarse-grained/val_coarse_grained.json'))
test_data = utils.build_data_from_jsonl(
    os.path.join(utils.DIRECTORY_NAME, '../../data/coarse-grained/test_coarse_grained.json'))
senses = utils.build_all_senses(os.path.join(utils.DIRECTORY_NAME,"../../data/map/coarse_fine_defs_map.json"))

#As vocabulary init class accept lists as input, this aux function does the conversion
list_candidates = utils.list_all_values(training_data,'candidates')
vocab = vocabulary.Vocabulary(training_data["words"],senses)
#print(vocab.labels_to_idx)
train_dataset = wsddataset.WsdDataset(training_data["words"],training_data["senses"],vocab.word_to_idx,vocab.labels_to_idx)
valid_dataset = wsddataset.WsdDataset(valid_data["words"],valid_data["senses"],vocab.word_to_idx,vocab.labels_to_idx)
test_dataset = wsddataset.WsdDataset(test_data["words"],test_data["senses"],vocab.word_to_idx,vocab.labels_to_idx)

train_dataloader = DataLoader(train_dataset,batch_size=utils.batch_size,collate_fn=utils.collate_fn,shuffle=False)
valid_dataloader = DataLoader(valid_dataset,batch_size=utils.batch_size,collate_fn=utils.collate_fn,shuffle=False)
test_dataloader = DataLoader(test_dataset,batch_size=utils.batch_size,collate_fn=utils.collate_fn,shuffle=False)

#print(train_dataset[150][0])
#utils.collate_fn(train_dataset[150])

# instantiate the model
model = mod.NERModule(utils.LANGUAGE_MODEL_NAME, len(vocab.labels_to_idx.keys()), fine_tune_lm=False)
model.to(utils.device)

# optimizer
groups = [
  {
    "params": model.classifier.parameters(),
    "lr": utils.learning_rate,
    "weight_decay": utils.weight_decay,
  },
  {
    "params": model.transformer_model.parameters(),
    "lr": utils.transformer_learning_rate,
    "weight_decay": utils.transformer_weight_decay,
  },
]
optimizer = Adam(groups)
train.train(train_dataloader,utils.epochs,optimizer,model,utils.device,valid_dataloader,vocab.labels_to_idx)