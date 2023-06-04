
import os
import vocabulary
import wsddataset
from torch.utils.data import DataLoader
import time
import model as mod
import pytorch_lightning as pl
import utilz
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


training_data = utilz.build_data_from_json(
    os.path.join(utilz.DIRECTORY_NAME, '../../data/coarse-grained/train_coarse_grained.json'))

valid_data = utilz.build_data_from_json(
    os.path.join(utilz.DIRECTORY_NAME, '../../data/coarse-grained/val_coarse_grained.json'))
#test_data = utils.build_data_from_json(
#    os.path.join(utils.DIRECTORY_NAME, '../../data/coarse-grained/test_coarse_grained.json'))

senses = utilz.build_all_senses(os.path.join(utilz.DIRECTORY_NAME,"../../data/map/coarse_fine_defs_map.json"))



vocab = vocabulary.Vocabulary(training_data["words"],senses)

train_dataset = wsddataset.WsdDataset(training_data["samples"],vocab.labels_to_idx)
valid_dataset = wsddataset.WsdDataset(valid_data["samples"],vocab.labels_to_idx)
#test_dataset = wsddataset.WsdDataset(test_data["words"],test_data["senses"],vocab.labels_to_idx)
train_dataloader = DataLoader(train_dataset,batch_size=utilz.BATCH_SIZE,collate_fn=utilz.collate_fn,shuffle=False, num_workers=utilz.NUM_WORKERS)
valid_dataloader = DataLoader(valid_dataset,batch_size=utilz.BATCH_SIZE,collate_fn=utilz.collate_fn,shuffle=False,num_workers=utilz.NUM_WORKERS)
#test_dataloader = DataLoader(test_dataset,batch_size=utils.BATCH_SIZE,collate_fn=utils.collate_fn,shuffle=False,num_workers=utilz.NUM_WORKERS)

model = mod.WSD(utilz.LANGUAGE_MODEL_NAME, len(vocab.labels_to_idx.keys()),vocab.idx_to_labels, fine_tune_lm=True)
logger = TensorBoardLogger(os.path.join(utilz.DIRECTORY_NAME,"lightning_logs/") , name="giacomo")
trainer = pl.Trainer(max_epochs = utilz.NUM_EPOCHS,callbacks=EarlyStopping(monitor="val_loss",patience=5),logger=logger)
trainer.fit(model,train_dataloader,valid_dataloader)

