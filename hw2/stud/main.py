import utils
import os
import vocabulary
import wsddataset
from torch.utils.data import DataLoader
import time
import model as mod
import pytorch_lightning as pl
from transformers import DataCollatorForTokenClassification


training_data = utils.build_data_from_jsonl(
    os.path.join(utils.DIRECTORY_NAME, '../../data/coarse-grained/train_coarse_grained.json'))
valid_data = utils.build_data_from_jsonl(
    os.path.join(utils.DIRECTORY_NAME, '../../data/coarse-grained/val_coarse_grained.json'))
test_data = utils.build_data_from_jsonl(
    os.path.join(utils.DIRECTORY_NAME, '../../data/coarse-grained/test_coarse_grained.json'))
senses = utils.build_all_senses(os.path.join(utils.DIRECTORY_NAME,"../../data/map/coarse_fine_defs_map.json"))

data_collator = DataCollatorForTokenClassification(tokenizer=utils.tokenizer)


vocab = vocabulary.Vocabulary(training_data["words"],senses)

train_dataset = wsddataset.WsdDataset(training_data["words"],training_data["senses"],vocab.word_to_idx,vocab.labels_to_idx)
valid_dataset = wsddataset.WsdDataset(valid_data["words"],valid_data["senses"],vocab.word_to_idx,vocab.labels_to_idx)
test_dataset = wsddataset.WsdDataset(test_data["words"],test_data["senses"],vocab.word_to_idx,vocab.labels_to_idx)

train_dataloader = DataLoader(train_dataset,batch_size=utils.batch_size,collate_fn=utils.collate_fn,shuffle=False)
valid_dataloader = DataLoader(valid_dataset,batch_size=utils.batch_size,collate_fn=utils.collate_fn,shuffle=False)
#test_dataloader = DataLoader(test_dataset,batch_size=utils.batch_size,collate_fn=utils.collate_fn,shuffle=False)

model = mod.WSD(utils.LANGUAGE_MODEL_NAME, len(vocab.labels_to_idx.keys()),vocab.idx_to_labels, fine_tune_lm=False)

trainer = pl.Trainer()
trainer.fit(model,train_dataloader,valid_dataloader)

