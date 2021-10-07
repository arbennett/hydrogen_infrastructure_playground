import mixins
import pytorch_lightning as pl
import torch
import xarray as xr
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union, List


class CNN2DDataset(Dataset):

    def __init__(self):
        super().__init__()

    def __len__(self):
        # The number of samples in the dataset
        pass

    def __getitem__(self, idx):
        # Get an individual sample
        pass


class CNN2DDataModule(pl.LightningDataModule, mixins.ScalerMixin):

    def __init__(
        self,
        pfidb_or_pfmetadata_file,
        input_vars: Optional[List[str]]=None,
        target_vars: Optional[List[str]]=None,
        scalers: Optional[Union[str, dict]]=None,
        batch_size=256
    ):
        super().__init__()
        self.pfidb_or_pfmetadata_file = pfidb_or_pfmetadata_file
        self.batch_size = batch_size


    def setup(self, stage: Optional[str] = None):

        if stage in (None, 'fit'):
            pass

        if stage in (None, 'train'):
            pass

        if stage in (None, 'test'):
            pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def teardown(self, stage: Optional[str] = None):
        pass

