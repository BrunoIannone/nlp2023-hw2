
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
from pytorch_lightning.profilers import PyTorchProfiler
import datamodule
import torch
training_data = utilz.build_data_from_json(
    os.path.join(utilz.DIRECTORY_NAME, '../../data/coarse-grained/train_coarse_grained.json'))

valid_data = utilz.build_data_from_json(
    os.path.join(utilz.DIRECTORY_NAME, '../../data/coarse-grained/val_coarse_grained.json'))
#test_data = utils.build_data_from_json(
#    os.path.join(utils.DIRECTORY_NAME, '../../data/coarse-grained/test_coarse_grained.json'))

senses = utilz.build_all_senses(os.path.join(utilz.DIRECTORY_NAME,"../../data/map/coarse_fine_defs_map.json"))

vocab = vocabulary.Vocabulary(training_data["words"],senses)
#print(vocab.idx_to_labels)
dm = datamodule.WsdDataModule(training_data,valid_data,vocab.labels_to_idx)

model = mod.WSD(utilz.LANGUAGE_MODEL_NAME, len(vocab.labels_to_idx.keys()),vocab.idx_to_labels, fine_tune_lm=True)
logger = TensorBoardLogger(os.path.join(utilz.DIRECTORY_NAME,"tb_logs"))
profiler = PyTorchProfiler(on_trace_ready = torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),trace_memory = True)
trainer = pl.Trainer(max_epochs = utilz.NUM_EPOCHS,callbacks=EarlyStopping(monitor="val_loss", patience=5),logger=logger, profiler=profiler)
#trainer = pl.Trainer(max_epochs = utilz.NUM_EPOCHS,logger=logger, profiler=profiler)

trainer.fit(model,datamodule = dm)
trainer.validate(model,datamodule=dm)

