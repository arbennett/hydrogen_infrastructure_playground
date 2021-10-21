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


Z_REDUCERS = {
        'mean': lambda x: x.mean(dim='z'),
        'median': lambda x: x.median(dim='z'),
        'sum': lambda x: x.sum(dim='z'),
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
        z_strategy: Union[str, int]=None,
        patch_sizes: Mapping[str, int]={},
        max_patches: int=None,
        dtype: torch.dtype=torch.float32,
        read_parallel: bool=False,
        dataset_chunks: Mapping[str, int]={'time': 1},
        raw_isel_args: Mapping[str, int]={}
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
                          read_outputs=read_vars,
                          chunks=dataset_chunks
            ).isel(raw_isel_args)
        elif isinstance(filename_or_obj, str):
            self.ds = xr.open_dataset(
                          filename_or_obj,
                          read_inputs=read_vars,
                          read_outputs=read_vars
            ).isel(raw_isel_args)
        elif isinstance(filename_or_obj, xr.Dataset):
            self.ds = filename_or_obj.isel(raw_isel_args)

        self.dtype = dtype
        self.z_strategy = z_strategy
        if patch_sizes:
            self.patch_sizes = patch_sizes
            self.patches = self._gen_patches()
            self.n_patches = len(self.patches)
        else:
            # Single patch that covers the entire domain
            self.patches = np.array([{'x': slice(0, None), 'y': slice(0, None)}])
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
        """
        NOTE: Untested! Just throwing some code in here so
        I don't lose track of it... to be used when patch_sizes
        is a valid input
        """
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
                _patch[k] = sub_patch[i][-1].astype(int).values
            #yield _patch  # <- TODO: Can we figure out how to make this a generator?
            patches.append(_patch)
        return np.array(patches)

    def __len__(self):
        # The number of samples in the dataset
        return (len(self.ds['time']) - 1) * len(self.patches)

    def _require_z_strategy(self):
        have_z_strategy = self.z_strategy is not None
        need_z_strategy = False
        for v in self.input_vars + self.target_vars:
            if 'z' in self.ds[v].dims:
                need_z_strategy = True
                break

        if need_z_strategy and not have_z_strategy:
            raise RuntimeError(
                    "You have a z dimension in your input/output variables, "
                    "but have not specified a valid z_strategy to handle this!")
        return need_z_strategy

    def _reduce_along_z(self, input_ds, output_ds):
        if self._require_z_strategy():
            if isinstance(self.z_strategy, int):
                if 'z' in input_ds.dims:
                    input_ds = input_ds.isel(z=self.z_strategy)
                if 'z' in output_ds.dims:
                    output_ds = output_ds.isel(z=self.z_strategy)
            elif isinstance(self.z_strategy, str):
                reducer = Z_REDUCERS[self.z_strategy]
                if 'z' in input_ds:
                    input_ds = reducer(input_ds)
                if 'z' in output_ds:
                    output_ds = reducer(output_ds)
        return input_ds, output_ds

    def _load_batch(self, time_idx):
        return self.ds.isel(time=[time_idx, time_idx+1]).load()

    def __getitem__(self, idx):
        # Get an individual sample
        patch_idx = idx % self.n_patches
        time_idx = idx // self.n_patches
        if not self.current_batch:
            self.current_batch = self._load_batch(time_idx)
        if time_idx != self.current_batch['time'].values[0]:
            self.current_batch = self._load_batch(time_idx)
        all_vars = self.input_vars + self.target_vars
        sub_ds = self.current_batch.isel({'time': 0, **self.patches[patch_idx]}).persist()
        input_ds = sub_ds[self.input_vars]
        output_ds = sub_ds[self.target_vars]
        input_ds, output_ds = self._reduce_along_z(input_ds, output_ds)

        # Scale as necessary
        X, y = [], []
        for v in self.input_vars:
            X.append(self.scalers[v](input_ds[v].values))
        for v in self.target_vars:
            y.append(self.scalers[v](output_ds[v].values))

        X = torch.from_numpy(np.stack(X)).type(self.dtype)
        y = torch.from_numpy(np.stack(y)).type(self.dtype)
        return X, y


class Conv2dDataModule(pl.LightningDataModule, mixins.ScalerMixin):

    def __init__(
        self,
        pfidb_or_pfmetadata_file: Union[List[str], str],
        in_vars: Union[List[str], str],
        out_vars: Union[List[str], str],
        z_strategy: Union[str, int]=None,
        patch_sizes: Mapping[str, int]={},
        raw_isel_args: Mapping[str, int]={},
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
        self.z_strategy = z_strategy
        self.patch_sizes = patch_sizes
        self.train_frac = train_frac
        self.train_size = train_size
        self.val_size = val_size
        self.pfidb_or_pfmetadata_file = pfidb_or_pfmetadata_file
        self.batch_size = batch_size
        self.max_patches = max_patches
        self.num_workers = num_workers
        self.raw_isel_args = raw_isel_args
        self._shape = None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            self._full = Conv2dDataset(
                    self.pfidb_or_pfmetadata_file,
                    in_vars=self.in_vars,
                    out_vars=self.out_vars,
                    z_strategy=self.z_strategy,
                    patch_sizes=self.patch_sizes,
                    max_patches=self.max_patches,
                    raw_isel_args=self.raw_isel_args
            )
            full_size = len(self._full)
            n_train = int(self.train_frac * full_size)
            n_val = full_size - n_train
            if not self.train_size:
                self.train_size = int(self.train_frac * full_size)
                self.val_size = full_size - self.train_size
            self.dataset_train, self.dataset_val = random_split(
                    self._full, [self.train_size, self.val_size])

        if stage in (None, 'test'):
            self.dataset_test = Conv2dDataset(
                    self.pfidb_or_pfmetadata_file,
                    in_vars=self.in_vars,
                    out_vars=self.out_vars,
                    z_strategy=self.z_strategy,
                    patch_sizes=self.patch_sizes,
                    max_patches=self.max_patches,
                    raw_isel_args=self.raw_isel_args
                    )


    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

    def teardown(self, stage: Optional[str] = None):
        pass

    @property
    def shape(self):
        if not self._shape:
            base_shape = {k: self.patch_sizes.get(k, None) for k in ('y','x')}
            if None in list(base_shape.values()):
                if isinstance(self.pfidb_or_pfmetadata_file, List):
                    f = self.pfidb_or_pfmetadata_file[0]
                else:
                    f = self.pfidb_or_pfmetadata_file
                with xr.open_dataset(
                    f, read_inputs=self.all_vars, read_outputs=self.all_vars
                ) as ds:
                    ds_shape = len(ds['y']), len(ds['x'])
                for v2, (k, v) in zip(ds_shape, base_shape.items()):
                    if v is None:
                        base_shape[k] = v2
            self._shape = tuple(base_shape[c] for c in ('y','x'))
        return self._shape

    @property
    def feature_names(self):
        return (self.in_vars, self.out_vars)
