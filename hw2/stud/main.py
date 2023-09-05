
import os
import vocabulary
import time
import wsd_model as mod
import pytorch_lightning as pl
import transformer_utils
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import datamodule
import glove
import glove_datamodule
import torch
import elmo_wsd
import elmo_datamodule
import rnn_utils as utils
from itertools import product
import tqdm
from termcolor import colored

CHOICE =  "elmo" # {"transformer", "glove", "elmo"}
GRAINE =  "fine"     # {"coarse", "fine"}

LOG_SAVE_DIR_NAME = "elmo/coarse_smoke_test"
CKPT_SAVE_DIR_NAME= "elmo/coarse_smoke_test"

print(colored(str(("CHOSEN ARCHITECTURE:", CHOICE, "CHOSEN GRAINE:", GRAINE)), "green"))

if CHOICE == "transformer":
    
    # Transformers hyperparameters combination
    hyp_comb = list(product(transformer_utils.LEARNING_RATE, transformer_utils.transformer_learning_rate, [0], transformer_utils.LIN_DROPOUT,transformer_utils.weight_decay,transformer_utils.transformer_weight_decay))

elif CHOICE == "glove":
    
    # GloVe hyperparameters combination
    hyp_comb = list(product(utils.LEARNING_RATE, utils.LSTM_LR, utils.DROPOUT_EMBED,utils.DROPOUT_LAYER,utils.LIN_WD,utils.LSTM_WD))

else: #CHOICE == "elmo"
    
    # ELMo hyperparameters combination
    hyp_comb = list(product(utils.LEARNING_RATE, utils.ELMO_LR, [0], utils.DROPOUT_LAYER, utils.LIN_WD,utils.ELMO_WD))

for hyperparameter in tqdm.tqdm(hyp_comb,colour="yellow", desc="Tried combinations"):
    
    lin_lr = hyperparameter[0]
    backbone_lr = hyperparameter[1]
    dropout = hyperparameter[2]
    lin_dropout = hyperparameter[3]
    lin_wd = hyperparameter[4]
    backbone_wd = hyperparameter[5]
    
    print(colored(str(("LIN_LR:", lin_lr,"BACKBONE_LR:", backbone_lr,"DROPOUT_EMBED:", dropout, "LINEAR_DROPOUT:", lin_dropout, "LINEAR_WD:", lin_wd, "BACKBONE_WD: ", backbone_wd)), "yellow"))
    
    if GRAINE == "coarse": #JSON COARSE DATA PROCESSING
                
        training_data = transformer_utils.build_data_from_json(
            os.path.join(transformer_utils.DIRECTORY_NAME, '../../data/coarse-grained/train_coarse_grained.json'))
        valid_data = transformer_utils.build_data_from_json(
            os.path.join(transformer_utils.DIRECTORY_NAME, '../../data/coarse-grained/val_coarse_grained.json'))
        test_data = transformer_utils.build_data_from_json(
            os.path.join(transformer_utils.DIRECTORY_NAME, '../../data/coarse-grained/test_coarse_grained.json'))
        
        print(colored("Built coarse data","green"))
        
    else:  #JSON FINE DATA PROCESSING

        training_data = transformer_utils.build_data_from_json(
            os.path.join(transformer_utils.DIRECTORY_NAME, '../../data/fine-grained/train_fine_grained.json'))
        valid_data = transformer_utils.build_data_from_json(
            os.path.join(transformer_utils.DIRECTORY_NAME, '../../data/fine-grained/test_fine_grained.json'))
        test_data = transformer_utils.build_data_from_json(
            os.path.join(transformer_utils.DIRECTORY_NAME, '../../data/fine-grained/val_fine_grained.json'))

        print(colored("Built fine data","green"))
        
    if GRAINE == 'coarse': ## Vocab and senses building for coarse
            
            senses = transformer_utils.build_all_senses(os.path.join(transformer_utils.DIRECTORY_NAME,"../../data/map/coarse_fine_defs_map.json"),fine_grained=False)
            vocab = vocabulary.Vocabulary(labels=senses,save_vocab=False)
            
            print(colored("Built coarse senses and vocab","green"))
            
    else: #GRAINE == "fine" ## Vocab and senses building for fine
        
        fine_senses = transformer_utils.build_all_senses(os.path.join(transformer_utils.DIRECTORY_NAME,"../../data/map/coarse_fine_defs_map.json"),fine_grained=True)
        fine_vocab = vocabulary.Vocabulary(labels=fine_senses,save_vocab=False)
        
        print(colored("Built senses and vocab for fine","green"))
    

    
    


    #senses = []
    #for sample in training_data['samples']:
    #    for sense in sample['senses']:
    #        senses.append(sample['senses'][sense]) 
    #print((senses))
    #time.sleep(10)

    #vocab = vocabulary.Vocabulary(senses,save_vocab=False)
    #print(len(vocab.labels_to_idx))

    ### MODEL RELATED INITIALIZATIONS ###

    logger = TensorBoardLogger(os.path.join(transformer_utils.DIRECTORY_NAME,"tb_logs/"+LOG_SAVE_DIR_NAME) + str(lin_lr) + ", " + str(backbone_lr) + ", " + str(dropout)+ ", " + str(lin_dropout)+ ", " + str(lin_wd) + ", " + str(backbone_wd))
    trainer = pl.Trainer(log_every_n_steps=50,max_epochs = transformer_utils.NUM_EPOCHS,callbacks=[EarlyStopping(monitor="val_loss", patience=5,mode='min'), ModelCheckpoint(filename= str(lin_lr) + ", " + str(backbone_lr) + ", " + str(dropout)+ ", " + str(lin_dropout)+ ", " + str(lin_wd) + ", " + str(backbone_wd),monitor='valid_f1',save_top_k=1,every_n_epochs=1,mode='max',save_weights_only=False,verbose=True,dirpath=os.path.join(transformer_utils.DIRECTORY_NAME,'../../model/' + CKPT_SAVE_DIR_NAME))],logger=logger,accelerator='gpu')
    
    print(colored("Built logger and trainer","green"))
    
    if CHOICE == "transformer": ### TRANSFORMERS ###

        if GRAINE == "coarse":
            
            tensor_dm = datamodule.WsdDataModule(training_data,valid_data,test_data,vocab.labels_to_idx)
            tensor_model = mod.WSD(transformer_utils.LANGUAGE_MODEL_NAME,len(vocab.labels_to_idx),fine_tune_lm=True,lin_lr=lin_lr,lin_dropout=lin_dropout,lin_wd=lin_wd,backbone_lr=backbone_lr,backbone_wd = backbone_wd)
            
            print(colored("Starting transformer coarse training...","green"))
            trainer.fit(tensor_model,datamodule = tensor_dm)
            
            print(colored("Starting transformer coarse testing...","green"))
            tensor_model.eval()
            trainer.test(tensor_model,datamodule = tensor_dm)#,ckpt_path="best")
        
        else:
            
            tensor_fine_dm = datamodule.WsdDataModule(training_data,valid_data,test_data,fine_vocab.labels_to_idx)
            fine_model = mod.WSD(transformer_utils.LANGUAGE_MODEL_NAME,len(fine_vocab.labels_to_idx),fine_tune_lm=True,lin_lr=lin_lr,lin_dropout=lin_dropout,lin_wd=lin_wd,backbone_lr=backbone_lr,backbone_wd = backbone_wd)
            
            print(colored("Starting transformer fine training...","green"))
            trainer.fit(fine_model,datamodule = tensor_fine_dm)
            
            print(colored("Starting transformer fine testing...","green"))
            fine_model.eval()
            trainer.test(fine_model,datamodule = tensor_fine_dm)#,ckpt_path="best")

    #model = mod.WSD.load_from_checkpoint(os.path.join(utilz.DIRECTORY_NAME, '../../model/0.001, 0.002, 0.01.ckpt'),map_location='cpu', strict=False)

    

    elif CHOICE == "glove": ### GLOVE ### 

        if GRAINE == "coarse": 
            
            glove_model = glove.Glove_WSD(utils.EMBEDDING_DIM, utils.HIDDEN_DIM,len(vocab.labels_to_idx), utils.LAYERS_NUM, lin_lr,backbone_lr,dropout, lin_dropout,lin_wd,backbone_wd)
            glove_dm = glove_datamodule.WsdDataModule(training_data,valid_data,test_data,vocab.labels_to_idx)
            
            print(colored("Starting glove coarse training...","green"))
            trainer.fit(glove_model,datamodule = glove_dm)
            
            print(colored("Starting glove coarse testing...","green"))
            glove_model.eval()
            trainer.test(glove_model,datamodule = glove_dm)#,ckpt_path="best")

        else:
            glove_model = glove.Glove_WSD(utils.EMBEDDING_DIM, utils.HIDDEN_DIM,len(fine_vocab.labels_to_idx), utils.LAYERS_NUM, lin_lr,backbone_lr,dropout, lin_dropout,lin_wd,backbone_wd)

            glove_fine_dm = glove_datamodule.WsdDataModule(training_data,valid_data,test_data,fine_vocab.labels_to_idx)
            
            print(colored("Starting glove fine training...","green"))
            trainer.fit(glove_model,datamodule = glove_fine_dm)

            print(colored("Starting glove fine testing...","green"))
            glove_model.eval()
            trainer.test(glove_model,datamodule = glove_fine_dm)#,ckpt_path="best")
    
    else: ### ELMO ###

        if GRAINE == "coarse":
            
            elmo_model = elmo_wsd.Elmo_WSD(utils.ELMO_HIDDEN_DIM,len(vocab.labels_to_idx),lin_lr,backbone_lr,dropout,lin_wd,backbone_wd)
            elmo_dm = elmo_datamodule.WsdDataModule(training_data,valid_data,test_data,vocab.labels_to_idx)
            
            print(colored("Starting elmo coarse training...","green"))
            trainer.fit(elmo_model,datamodule = elmo_dm)
            
            print(colored("Starting elmo coarse testing...","green"))
            elmo_model.eval()
            trainer.test(elmo_model,datamodule = elmo_dm)#,ckpt_path="best")
        
        else:
            
            elmo_model = elmo_wsd.Elmo_WSD(utils.ELMO_HIDDEN_DIM,len(fine_vocab.labels_to_idx),lin_lr,backbone_lr,dropout,lin_wd,backbone_wd)
            elmo_fine_dm = elmo_datamodule.WsdDataModule(training_data,valid_data,test_data,fine_vocab.labels_to_idx)
            
            print(colored("Starting elmo fine training...","green"))
            trainer.fit(elmo_model,datamodule = elmo_fine_dm)
            
            print(colored("Starting elmo fine testing...","green"))
            elmo_model.eval()
            trainer.test(elmo_model,datamodule = elmo_fine_dm)#,ckpt_path="best")


    
    

print(colored("Pipeline over","red"))