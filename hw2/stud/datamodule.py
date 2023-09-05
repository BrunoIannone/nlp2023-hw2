from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import transformer_utils
import lstm_utils
import wsddataset

class WsdDataModule(LightningDataModule):
    """Datamodule for WSD with tensor

    
    """
    def __init__(self,training_data:dict, valid_data:dict, test_data:dict,labels_to_idx:dict):
        """Init function for WSD datamodule with tensor

        Args:
            training_data (dict): {sample:{List[sample_dicts]}} for training
            valid_data (dict): {sample:{List[sample_dicts]}} for valid
            test_data (dict): {sample:{List[sample_dicts]}} for test
            labels_to_idx (dict):  dictionary with structure {label:index} 
        """
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
            batch_size = transformer_utils.BATCH_SIZE,
            num_workers = transformer_utils.NUM_WORKERS,
            shuffle = False,
            collate_fn=transformer_utils.collate_fn
        ) 
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size = transformer_utils.BATCH_SIZE,
            num_workers = transformer_utils.NUM_WORKERS,
            shuffle = False,
            collate_fn=transformer_utils.collate_fn
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = transformer_utils.BATCH_SIZE,
            num_workers = transformer_utils.NUM_WORKERS,
            shuffle = False,
            collate_fn=transformer_utils.collate_fn
        )