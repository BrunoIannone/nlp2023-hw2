
import os
import vocabulary
import wsddataset
from torch.utils.data import DataLoader
import time
import wsd_model as mod
import pytorch_lightning as pl
import utilz
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
import datamodule
import torch
training_data = utilz.build_data_from_json(
    os.path.join(utilz.DIRECTORY_NAME, '../../data/coarse-grained/train_coarse_grained.json'))

valid_data = utilz.build_data_from_json(
    os.path.join(utilz.DIRECTORY_NAME, '../../data/coarse-grained/val_coarse_grained.json'))
test_data = utilz.build_data_from_json(
    os.path.join(utilz.DIRECTORY_NAME, '../../data/coarse-grained/test_coarse_grained.json'))

senses = utilz.build_all_senses(os.path.join(utilz.DIRECTORY_NAME,"../../data/map/coarse_fine_defs_map.json"))

vocab = vocabulary.Vocabulary(training_data["words"],senses,save_vocab=True)
#vocab = vocabulary.Vocabulary(training_data["words"],training_data['labels'])

#print(vocab.idx_to_labels)
dm = datamodule.WsdDataModule(training_data,valid_data,test_data,vocab.labels_to_idx)

#model = mod.WSD(utilz.LANGUAGE_MODEL_NAME, len(vocab.labels_to_idx.keys()),vocab.idx_to_labels, fine_tune_lm=True)
model = mod.WSD.load_from_checkpoint(os.path.join(utilz.DIRECTORY_NAME, 'glossbert2_0.884.ckpt'),map_location=utilz.DEVICE)
logger = TensorBoardLogger(os.path.join(utilz.DIRECTORY_NAME,"tb_logs"))
#profiler = PyTorchProfiler(on_trace_ready = torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),trace_memory = True)
trainer = pl.Trainer(max_epochs = utilz.NUM_EPOCHS,callbacks=[EarlyStopping(monitor="val_loss", patience=5,mode='min'), ModelCheckpoint(monitor='valid_f1',save_top_k=1,every_n_epochs=1,mode='max',save_weights_only=True,dirpath='/home/bruno/Desktop/nlp2023-hw2/hw2/stud')],logger=logger)
#trainer = pl.Trainer(max_epochs = utilz.NUM_EPOCHS,logger=logger, profiler=profiler)

#trainer.fit(model,datamodule = dm)
#trainer.validate(model,datamodule=dm)
trainer.test(model,datamodule = dm)
