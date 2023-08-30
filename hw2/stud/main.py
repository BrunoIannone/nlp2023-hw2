
import os
import vocabulary
import time
import wsd_model as mod
import pytorch_lightning as pl
import utilz
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, BackboneFinetuning, BatchSizeFinder, LearningRateFinder
from pytorch_lightning.loggers import TensorBoardLogger
import datamodule
import lstm_datamodule
import torch
import elmo_wsd
import lstm_utils as utils
import glove
from itertools import product
import tqdm

#hyp_comb = list(product(utils.LEARNING_RATE, utils.LSTM_LR,utils.DROPOUT_EMBED,utils.DROPOUT_LAYER, utils.LIN_WD,utils.LSTM_WD))
hyp_comb = list(product(utilz.LEARNING_RATE, utilz.transformer_learning_rate, utilz.LIN_DROPOUT,utilz.weight_decay,utilz.transformer_weight_decay))

#print(hyp_comb)

for hyperparameter in tqdm.tqdm(hyp_comb):
    
    print(hyperparameter)
    lin_lr = hyperparameter[0]
    backbone_lr = hyperparameter[1]
    dropout =0# hyperparameter[2]
    lin_dropout = hyperparameter[2]
    lin_wd = hyperparameter[3]
    backbone_wd = hyperparameter[4]

    
    #JSON DATA PROCESSING
    training_data = utilz.build_data_from_json(
        os.path.join(utilz.DIRECTORY_NAME, '../../data/fine-grained/train_fine_grained.json'))
    valid_data = utilz.build_data_from_json(
        os.path.join(utilz.DIRECTORY_NAME, '../../data/fine-grained/test_fine_grained.json'))
    test_data = utilz.build_data_from_json(
        os.path.join(utilz.DIRECTORY_NAME, '../../data/fine-grained/val_fine_grained.json'))

    #BUILD LABEL VOCABULARY
    senses = utilz.build_all_senses(os.path.join(utilz.DIRECTORY_NAME,"../../data/map/coarse_fine_defs_map.json"),fine_grained=True)
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

    #dm = lstm_datamodule.WsdDataModule(training_data,valid_data,test_data,vocab.labels_to_idx)

    #model = glove.Glove_WSD(utils.EMBEDDING_DIM, utils.HIDDEN_DIM,len(vocab.labels_to_idx), utils.LAYERS_NUM, lin_lr,lstm_lr,dropout, lin_dropout,lin_wd,lstm_wd)
    model = mod.WSD(utilz.LANGUAGE_MODEL_NAME,len(vocab.labels_to_idx.keys()),vocab.idx_to_labels,fine_tune_lm=True,lin_lr=lin_lr,lin_dropout=lin_dropout,lin_wd=lin_wd,backbone_lr=backbone_lr,backbone_wd = backbone_wd)
    #model = mod.WSD.load_from_checkpoint(os.path.join(utilz.DIRECTORY_NAME, '../../model/0.001, 0.002, 0.01.ckpt'),map_location='cpu', strict=False)
    #logger = TensorBoardLogger(os.path.join(utilz.DIRECTORY_NAME,"tb_logs/lstm_elmo") + str(utilz.LEARNING_RATE) + ", " + str(utilz.weight_decay) + ", " + str(utilz.transformer_weight_decay)))
    logger = TensorBoardLogger(os.path.join(utilz.DIRECTORY_NAME,"tb_logs/fine_grained/fine_grained") + str(lin_lr) + ", " + str(backbone_lr) + ", " + str(dropout)+ ", " + str(lin_dropout)+ ", " + str(lin_wd) + ", " + str(backbone_wd)+ ", " + str(utils.DROPOUT_EMBED))

    #trainer = pl.Trainer(max_epochs = utilz.NUM_EPOCHS,callbacks=[BackboneFinetuning(unfreeze_backbone_at_epoch=5,lambda_func=lambda epoch: 1,backbone_initial_lr=1e-5,initial_denom_lr=1,should_align=False,train_bn=True,verbose=True),EarlyStopping(monitor="val_loss", patience=5,mode='min'), ModelCheckpoint(filename= str(utilz.LEARNING_RATE) + ", " + str(utilz.weight_decay),monitor='valid_f1',save_top_k=1,every_n_epochs=1,mode='max',save_weights_only=False,verbose=True,dirpath=os.path.join(utilz.DIRECTORY_NAME,'../../model/finetune/glossbert/nowarmup_weightdecay'))],logger=logger,accelerator='gpu')#,auto_lr_find=True, auto_scale_batch_size=False)
    #trainer = pl.Trainer(max_epochs = utilz.NUM_EPOCHS,callbacks=[EarlyStopping(monitor="val_loss", patience=5,mode='min'), ModelCheckpoint(filename= str(utilz.LEARNING_RATE) + ", " + str(utilz.weight_decay)+ ", " + str(utilz.transformer_weight_decay),monitor='valid_f1',save_top_k=1,every_n_epochs=1,mode='max',save_weights_only=False,verbose=True,dirpath=os.path.join(utilz.DIRECTORY_NAME,'../../model/glossbert)'))],logger=logger,accelerator='gpu')#,auto_lr_find=True, auto_scale_batch_size=False)
    trainer = pl.Trainer(log_every_n_steps=21,max_epochs = utilz.NUM_EPOCHS,callbacks=[EarlyStopping(monitor="val_loss", patience=5,mode='min'), ModelCheckpoint(filename= str(lin_lr) + ", " + str(backbone_lr) + ", " + str(dropout)+ ", " + str(lin_dropout)+ ", " + str(lin_wd) + ", " + str(backbone_wd)+ ", " + str(utils.DROPOUT_EMBED) ,monitor='valid_f1',save_top_k=1,every_n_epochs=1,mode='max',save_weights_only=False,verbose=True,dirpath=os.path.join(utilz.DIRECTORY_NAME,'../../model/fine_grained'))],logger=logger,accelerator='gpu')#,auto_lr_find=True, auto_scale_batch_size=False)

    #START TRAINING ROUTINE
    #trainer.tune(model,datamodule=dm)
    trainer.fit(model,datamodule = dm)
    #model.eval()
    #trainer.validate(model,datamodule=dm)
    model.eval()
    trainer.test(model,datamodule = dm)#,ckpt_path="best")
