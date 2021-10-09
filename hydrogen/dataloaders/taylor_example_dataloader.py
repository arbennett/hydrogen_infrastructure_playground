import itertools
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union, List

from . import mixins

# NOTE: Base feature ranges from the Taylor river! Will need to be
# NOTE: changed when we go to CONUS!
SCALING = {
    'precipitation': ((0.0,2.3250430690836765),(0,1)),
    'temperature': ((243.04002434890052,294.7861684163411),(0,1)),
    'wind velocity': ((-9.555846213189705,15.570362051328024),(-1,1)),
    'UGRD': ((-9.555846213189705,15.570362051328024),(-1,1)),
    'VGRD': ((-9.459658953082725,10.973674557960384),(-1,1)),
    'atmospheric pressure': ((62042.070149739586,75857.17057291667),(0,1)),
    'specific humidity': ((0.00034727433576045206,0.010973041411489248),(0,1)),
    'downward shortwave radiation': ((44.300002892812095,341.0555042037224),(0,1)),
    'downward longwave radiation': ((93.16834259771652,356.44537989298504),(0,1)),
    'saturation': ((0.006598270240534006,1.0),(0,1)),
    'pressure': ((-12.673498421769821,53.29893832417906),(0,1)),
    'soil_moisture': ((0.0025535305830866597,0.48199999999999993),(0,1)),
    'wtd': ((0.0,54.86268596956115),(0,1)),
    'eflx_lh_tot': ((-11.987887556301255,227.2745242502459),(-1,1)),
    'eflx_lwrad_out': ((188.73408048992417,428.3168634458776),(0,1)),
    'eflx_sh_tot': ((-212.63476598064483,231.30395973560096),(0,1)),
    'eflx_soil_grnd': ((-225.05620842421095,190.92417048181622),(0,1)),
    'qflx_evap_tot': ((-0.017156259474353067,0.3255699677768812),(-1,1)),
    'qflx_evap_grnd': ((0.0,0.14114173688490758),(0,1)),
    'qflx_evap_soi': ((-0.03406682732982543,0.14114173688490758),(-1,1)),
    'qflx_evap_veg': ((-0.017161818440162336,0.3219445254210888),(-1,1)),
    'qflx_tran_veg': ((0.0,0.1636195226512655),(0,1)),
    'qflx_infl': ((-0.14098597960181578,1.7733137195552644),(-1,0)),
    'swe_out': ((0.0,754.746199964657),(0,1)),
    't_grnd': ((239.6187892890265,298.3340490161149),(0,1)),
    't_soil': ((273.1600036621094,298.3340490161149),(0,1)),
    'computed_porosity': ((0.33,0.482),(0,1)),
    'porosity': ((0.33,0.482),(0,1)),
    'slope_x': ((-0.40505003929138184,0.4567600190639496),(-1,1)),
    'slope_y': ((-0.3405400514602661,0.46806982159614563),(-1,1)),
    'computed_permeability': ((0.004675077,0.06),(0,1)),
    'permeability': ((0.004675077,0.06),(0,1))
    # 'flow':((0.0,538353.0964431178),(0,1))
}

class Conv2dDataset(Dataset, mixins.ScalerMixin):
    """
    TODO: Move the scale transforms to the ScalerMixin for real
    """

    def __init__(
        self,
        filename_or_obj: Union[List[str], str, xr.Dataset],
        in_vars: Union[List[str], str],
        out_vars: Union[List[str], str],
        # patch_sizes: Mapping[str, int]={}, # TODO: This will allow for patching out samples
        dtype: torch.dtype=torch.float32,
        read_parallel: bool=False
    ):
        super().__init__()
        if isinstance(filename_or_obj, List):
            self.ds = xr.open_mfdataset(
                    filename_or_obj, concat_dim='time', parallel=read_parallel)
        elif isinstance(filename_or_obj, str):
            self.ds = xr.open_dataset(filename_or_obj)
        elif isinstance(filename_or_obj, xr.Dataset):
            self.ds = filename_or_obj

        self.input_vars = in_vars if isinstance(in_vars, List) else [in_vars]
        self.target_vars = out_vars if isinstance(out_vars, List) else [out_vars]
        self.dtype = dtype
        self._gen_scalers()

    def _gen_scalers(self):
        """TODO: Make work for nd-arrays"""
        self.scalers = {}
        for v in self.input_vars + self.target_vars:
            in_range, out_range = SCALING[v]
            scaler = MinMaxScaler(feature_range=out_range)
            scaler.min_, scaler.max_ = in_range
            scaler.scale_ = in_range[1] - in_range[0]
            self.scalers[v] = scaler

    def _gen_patches(self):
        """
        NOTE: Untested! Just throwing some code in here so
        I don't lose track of it... to be used when patch_sizes
        is a valid input
        """
        patch_binner = {}
        patch_groups = {}
        for k, v in self.patch_sizes.items():
            patch_binner[k] = self.ds[k].isel(slice(0, None, v))
            patch_groups[k] = self.ds[k].groupby_bins(k, patch_binner[k])

        # Could this be more efficient?
        patches = []
        for sub_patch in itertools.product(*patch_groups.values()):
            _patch = {}
            for i, k in enumerate(self.patch_sizes.keys()):
                _patch[k] = sub_patch[i][-1]
            patches.append(_patch)
        return patches

    def __len__(self):
        # The number of samples in the dataset
        return len(self.ds['time']) - 1


    def __getitem__(self, idx):
        # Get an individual sample
        all_vars = self.input_vars + self.target_vars
        input_ds = self.ds.isel(time=idx)[self.input_vars]
        output_ds = self.ds.isel(time=idx+1)[self.target_vars]

        # Scale as necessary
        X, y = [], []
        for v in self.input_vars:
            X.append(self.scalers[v].transform(input_ds[v].values))

        for v in self.target_vars:
            y.append(self.scalers[v].transform(output_ds[v].values))

        X = torch.from_numpy(np.hstack(X), dtype=self.dtype)
        y = torch.from_numpy(np.hstack(y), dtype=self.dtype)
        return X, y


class Conv2dDataModule(pl.LightningDataModule, mixins.ScalerMixin):

    def __init__(
        self,
        pfidb_or_pfmetadata_file: Union[List[str], str],
        in_vars: Union[List[str], str],
        out_vars: Union[List[str], str],
        scalers: Optional[Union[str, dict]]=None,
        batch_size: int=256
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

    @property
    def shape(self):
        #TODO
        return (10, 10)

    @property
    def feature_names(self):
        #TODO
        return ([], [])
