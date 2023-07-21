
import os
import vocabulary
import time
import wsd_model as mod
import pytorch_lightning as pl
import utilz
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
import datamodule
import torch
import lstm
import lstm_utils as utils


#JSON DATA PROCESSING
training_data = utilz.build_data_from_json(
    os.path.join(utilz.DIRECTORY_NAME, '../../data/coarse-grained/train_coarse_grained.json'),save_words=True)
valid_data = utilz.build_data_from_json(
    os.path.join(utilz.DIRECTORY_NAME, '../../data/coarse-grained/val_coarse_grained.json'))
test_data = utilz.build_data_from_json(
    os.path.join(utilz.DIRECTORY_NAME, '../../data/coarse-grained/test_coarse_grained.json'))

#BUILD LABEL VOCANULARY
senses = utilz.build_all_senses(os.path.join(utilz.DIRECTORY_NAME,"../../data/map/coarse_fine_defs_map.json"))
vocab = vocabulary.Vocabulary(labels=senses,save_vocab=False)

#senses = []
#for sample in training_data['samples']:
#    for sense in sample['senses']:
#        senses.append(sample['senses'][sense]) 
#print((senses))
#time.sleep(10)

#vocab = vocabulary.Vocabulary(senses,save_vocab=False)
#print(len(vocab.labels_to_idx))

#MODEL RELATED INITIALIZATIONS

dm = datamodule.WsdDataModule(training_data,valid_data,test_data,vocab.labels_to_idx)
model = lstm.Lstm_WSD(utils.EMBEDDING_DIM, utils.HIDDEN_DIM,
                          50265, len(vocab.labels_to_idx), utils.LAYERS_NUM, None,vocab.idx_to_labels)
#model = mod.WSD(utilz.LANGUAGE_MODEL_NAME,utilz.LANGUAGE_MODEL_NAME_POS, len(vocab.labels_to_idx.keys()),vocab.idx_to_labels, fine_tune_lm=True)
#model = mod.WSD.load_from_checkpoint(os.path.join(utilz.DIRECTORY_NAME, '../../model/epoch=14-step=23145.ckpt'),map_location='cpu')
logger = TensorBoardLogger(os.path.join(utilz.DIRECTORY_NAME,"tb_logs/lstm"))
#profiler = PyTorchProfiler(on_trace_ready = torch.profiler.tensorboard_trace_handler(os.path.join(utilz.DIRECTORY_NAME,"tb_logs/profiler0")),trace_memory = True, schedule = torch.profiler.schedule(skip_first=10,wait=1,warmup=1,active=20))
trainer = pl.Trainer(max_epochs = utilz.NUM_EPOCHS,callbacks=[EarlyStopping(monitor="val_loss", patience=5,mode='min'), ModelCheckpoint(monitor='valid_f1',save_top_k=1,every_n_epochs=1,mode='max',save_weights_only=False,verbose=True,dirpath=os.path.join(utilz.DIRECTORY_NAME,'../../model/lstm'))],logger=logger,accelerator='gpu')
#trainer = pl.Trainer(max_epochs = utilz.NUM_EPOCHS,logger=logger, profiler=profiler)

#START TRAINING ROUTINE
trainer.fit(model,datamodule = dm)
#trainer.validate(model,datamodule=dm)
trainer.test(model,datamodule = dm)
#model = mod.WSD.load_from_checkpoint
