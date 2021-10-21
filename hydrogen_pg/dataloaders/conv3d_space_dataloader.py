import itertools
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from hydrogen.transform import float32_clamp_scaling
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Union, List, Mapping

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


class Conv3dDataset(Dataset, mixins.ScalerMixin):
    """
    TODO: Move the scale transforms to the ScalerMixin for real
    """

    def __init__(
        self,
        filename_or_obj: Union[List[str], str, xr.Dataset],
        in_vars: Union[List[str], str],
        out_vars: Union[List[str], str],
        patch_sizes: Mapping[str, int]={},
        dtype: torch.dtype=torch.float32,
        read_parallel: bool=False,
        max_patches: int=None
    ):
        super().__init__()
        self.input_vars = in_vars if isinstance(in_vars, List) else [in_vars]
        self.target_vars = out_vars if isinstance(out_vars, List) else [out_vars]
        self.all_vars = self.input_vars + self.target_vars
        read_vars = self._disambiguate_component_vars(self.all_vars)
        if isinstance(filename_or_obj, List):
            self.ds = xr.open_mfdataset(
                          filename_or_obj,
                          combine='nested',
                          concat_dim='time',
                          parallel=read_parallel,
                          read_inputs=read_vars,
                          read_outputs=read_vars
            )
        elif isinstance(filename_or_obj, str):
            self.ds = xr.open_dataset(
                          filename_or_obj,
                          read_inputs=read_vars,
                          read_outputs=read_vars
            )
        elif isinstance(filename_or_obj, xr.Dataset):
            self.ds = filename_or_obj

        self.dtype = dtype
        if patch_sizes:
            self.patch_sizes = patch_sizes
            self.patches = self._gen_patches()
            self.n_patches = len(self.patches)
        else:
            # Single patch that covers the entire domain
            # TODO: Handle z?
            self.patches = [{'x': slice(0, None), 'y': slice(0, None)}]
            self.n_patches = 1
        if max_patches:
            self.n_patches = np.min([max_patches, self.n_patches])
            self.patches = self.patches[0:self.n_patches]
        self._gen_scalers()
        self.current_batch = None

    def _disambiguate_component_vars(self, var_list):
        return list(set([v.split('_')[0] for v in var_list]))

    def _gen_scalers(self):
        """TODO: Make an interface which is serializable like sklearn"""
        self.scalers = {}
        for v in self.input_vars + self.target_vars:
            src_range, dst_range = SCALING[v]
            #scaler = MinMaxScaler(feature_range=out_range)
            #scaler.min_, scaler.max_ = in_range
            #scaler.scale_ = in_range[1] - in_range[0]
            self.scalers[v] = float32_clamp_scaling(src_range, dst_range)

    def _gen_patches(self):
        patch_binner = {}
        patch_groups = {}
        for k, v in self.patch_sizes.items():
            patch_binner[k] = self.ds[k].isel({k: slice(0, None, v)})
            patch_groups[k] = self.ds[k].groupby_bins(k, patch_binner[k])

        # Could this be more efficient?
        patches = []
        for sub_patch in itertools.product(*patch_groups.values()):
            _patch = {}
            for i, k in enumerate(self.patch_sizes.keys()):
                _patch[k] = sub_patch[i][-1].astype(int)
            #yield _patch  # <- TODO: Can we figure out how to make this a generator?
            patches.append(_patch)
        return patches

    def __len__(self):
        # The number of samples in the dataset
        return len(self.ds['time']) * len(self.patches)

    def _load_batch(self, time_idx):
        return self.ds.isel(time=[time_idx]).load()

    def __getitem__(self, idx):
        # Get an individual sample

        if not self.current_batch:
            self.current_batch = self._load_batch(time_idx)
        if time_idx != self.current_batch['time'].values[0]:
            self.current_batch = self._load_batch(time_idx)

        patch_idx = idx % self.n_patches
        time_idx = idx // self.n_patches
        input_ds = self.current_batch.isel(
                       {'time': time_idx, **self.patches[patch_idx]}
        )[self.input_vars]
        output_ds = self.current_batch.isel(
                        {'time': time_idx, **self.patches[patch_idx]}
        )[self.target_vars]

        # Scale as necessary
        X, y = [], []
        for v in self.input_vars:
            X.append(self.scalers[v](input_ds[v].values))
        for v in self.target_vars:
            y.append(self.scalers[v](output_ds[v].values))

        X = torch.from_numpy(np.stack(X)).type(self.dtype)
        y = torch.from_numpy(np.stack(y)).type(self.dtype)
        return X, y


class Conv3dDataModule(pl.LightningDataModule, mixins.ScalerMixin):

    def __init__(
        self,
        pfidb_or_pfmetadata_file: Union[List[str], str],
        in_vars: Union[List[str], str],
        out_vars: Union[List[str], str],
        patch_sizes: Mapping[str, int]={},
        train_frac: float=0.7,
        train_size: int=None,
        val_size: int=None,
        batch_size: int=256,
        max_patches: int=None,
        num_workers: int=0,
    ):
        super().__init__()
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.all_vars = in_vars + out_vars
        self.patch_sizes = patch_sizes
        self.train_frac = train_frac
        self.train_size = train_size
        self.val_size = val_size
        self.pfidb_or_pfmetadata_file = pfidb_or_pfmetadata_file
        self.batch_size = batch_size
        self.max_patches = max_patches
        self.num_workers = num_workers
        self._shape = None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            self._full = Conv3dDataset(
                    self.pfidb_or_pfmetadata_file,
                    patch_sizes=self.patch_sizes,
                    in_vars=self.in_vars,
                    out_vars=self.out_vars,
                    max_patches=self.max_patches
            )
            full_size = len(self._full)
            if not self.train_size:
                self.train_size = int(self.train_frac * full_size)
                self.val_size = full_size - self.train_size
            self.dataset_train, self.dataset_val = random_split(
                    self._full, [self.train_size, self.val_size])


        if stage in (None, 'test'):
            self.dataset_test = Conv3dDataset(
                    self.pfidb_or_pfmetadata_file,
                    patch_sizes=self.patch_sizes,
                    in_vars=self.in_vars,
                    out_vars=self.out_vars
            )


    def train_dataloader(self):
        return DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
                self.dataset_test,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
        )

    def teardown(self, stage: Optional[str] = None):
        pass

    @property
    def shape(self):
        if not self._shape:
            base_shape = {k: self.patch_sizes.get(k, None) for k in ('z','y','x')}
            if None in list(base_shape.values()):
                # Need to open file to tell
                if isinstance(self.pfidb_or_pfmetadata_file, List):
                    f = self.pfidb_or_pfmetadata_file[0]
                else:
                    f = self.pfidb_or_pfmetadata_file
                with xr.open_dataset(
                    f, read_inputs=self.all_vars, read_outputs=self.all_vars
                ) as ds:
                     ds_shape = len(ds['z']), len(ds['y']), len(ds['x'])
                for v2, (k, v) in zip(ds_shape, base_shape.items()):
                    if v is None:
                        base_shape[k] = v2
            self._shape = tuple(base_shape[c] for c in ('z','y','x'))
        return self._shape

    @property
    def feature_names(self):
        return (self.in_vars, self.out_vars)
