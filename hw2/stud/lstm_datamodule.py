from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import lstm_utils
import wsddataset

class WsdDataModule(LightningDataModule):
    def __init__(self,training_data, valid_data, test_data,labels_to_idx):
        super().__init__()
    
        self.training_data = training_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.labels_to_idx = labels_to_idx

    def setup(self, stage: str):
        
        self.train_dataset = wsddataset.WsdDataset(self.training_data["samples"],self.labels_to_idx)
        self.valid_dataset = wsddataset.WsdDataset(self.valid_data["samples"],self.labels_to_idx)
        self.test_dataset = wsddataset.WsdDataset(self.test_data["samples"],self.labels_to_idx)

    def train_dataloader(self):
        
        return DataLoader(
            self.train_dataset,
            batch_size = lstm_utils.BATCH_SIZE,
            num_workers = lstm_utils.NUM_WORKERS,
            shuffle = False,
            collate_fn=lstm_utils.collate_fn_glove
        ) 
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size = lstm_utils.BATCH_SIZE,
            num_workers = lstm_utils.NUM_WORKERS,
            shuffle = False,
            collate_fn=lstm_utils.collate_fn_glove
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = lstm_utils.BATCH_SIZE,
            num_workers = lstm_utils.NUM_WORKERS,
            shuffle = False,
            collate_fn=lstm_utils.collate_fn_glove
        )