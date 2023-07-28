
import os
import vocabulary
import time
import wsd_model as mod
import pytorch_lightning as pl
import utilz
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, BackboneFinetuning, BatchSizeFinder, LearningRateFinder
from pytorch_lightning.loggers import TensorBoardLogger
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


#model = lstm.Lstm_WSD(utils.EMBEDDING_DIM, utils.HIDDEN_DIM,
#                          50265, len(vocab.labels_to_idx), utils.LAYERS_NUM, None,vocab.idx_to_labels)
model = mod.WSD(utilz.LANGUAGE_MODEL_NAME,len(vocab.labels_to_idx.keys()),vocab.idx_to_labels,fine_tune_lm=False)
#model.load_from_checkpoint(os.path.join(utilz.DIRECTORY_NAME, '../../model/test/only_classifier.ckpt'),map_location='cpu')
logger = TensorBoardLogger(os.path.join(utilz.DIRECTORY_NAME,"tb_logs/test"))

trainer = pl.Trainer(max_epochs = utilz.NUM_EPOCHS,callbacks=[BackboneFinetuning(unfreeze_backbone_at_epoch=7,lambda_func=lambda epoch: 1,backbone_initial_lr=1e-5,initial_denom_lr=1,should_align=False,train_bn=True,verbose=True),EarlyStopping(monitor="val_loss", patience=5,mode='min'), ModelCheckpoint(monitor='valid_f1',save_top_k=1,every_n_epochs=1,mode='max',save_weights_only=False,verbose=True,dirpath=os.path.join(utilz.DIRECTORY_NAME,'../../model/test'))],logger=logger,accelerator='gpu')#,auto_lr_find=True, auto_scale_batch_size=False)

#START TRAINING ROUTINE
#trainer.tune(model,datamodule=dm)
trainer.fit(model,datamodule = dm)
#trainer.validate(model,datamodule=dm)
trainer.test(model,datamodule = dm)

