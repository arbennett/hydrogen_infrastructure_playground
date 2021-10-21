import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Union, List, Mapping

class DenseNet(pl.LightningModule):

    def __init__(
            self,
            grid_size=[25, 25],
            in_vars=List[str],
            out_vars=List[str],
            verbose=False,
            width=None,
            depth=6,
    ):
        super().__init__()
        self.use_dropout = False
        self.grid_size = grid_size
        self.flat_size = np.prod(grid_size)
        self.verbose = verbose
        self.in_vars = in_vars
        self.out_vars = out_vars

        self.width = width
        self.depth = depth
        self.activation = nn.LeakyReLU
        if not self.width:
            self.width = int(2 * np.prod(grid_size))

        Cin = len(in_vars)
        Hin = grid_size[0]
        Win = grid_size[1]
        self.in_size = len(self.in_vars) * self.flat_size
        self.out_size = len(self.out_vars) * self.flat_size

        # Initialize input layers
        layers = [
            nn.Flatten(),
            nn.Linear(self.in_size, self.width),
            self.activation()]
        # Initialize hidden layers
        for i in range(self.depth):
            layers.append(nn.Linear(self.width, self.width))
            layers.append(self.activation())
        # Initialize output layers
        layers.append(nn.Linear(self.width, self.out_size))
        layers.append(self.activation())
        self.module_list = nn.ModuleList(layers)

    def forward(self, x):
        for f in self.module_list:
            x = f(x)
        return x.view((-1, *self.grid_size))

    def configure_optimizers(self, opt=torch.optim.Adam, **kwargs):
        optimizer = opt(self.module_list.parameters(), **kwargs)
        return optimizer

    def configure_loss(self, loss_fun=F.mse_loss):
        self.loss_fun = loss_fun

    def training_step(self, train_batch, train_batch_idx):
        x, y = train_batch
        for f in self.module_list:
            x = f(x)
        y_hat = x.view((-1, *self.grid_size))
        loss = self.loss_fun(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        x, y = val_batch
        for f in self.module_list:
            x = f(x)
        y_hat = x.view((-1, *self.grid_size))
        loss = self.loss_fun(y_hat, y)
        self.log('val_loss', loss)
        return loss

    @property
    def shape(self):
        return self.grid_size

    @property
    def feature_names(self):
        return self.in_vars, self.out_vars
