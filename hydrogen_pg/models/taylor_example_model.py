import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Union, List, Mapping

class RMM_NN_2D_B1(pl.LightningModule):

    def __init__(
            self,
            grid_size=[25, 25],
            in_vars=List[str],
            out_vars=List[str],
            verbose=False,
    ):
        super().__init__()
        self.use_dropout = False
        self.grid_size = grid_size
        self.verbose = verbose
        self.in_vars = in_vars
        self.out_vars = out_vars

        # Inputs Parameters
        Cin = len(in_vars)
        Hin = grid_size[0]
        Win = grid_size[1]
        in_shape = [Cin, Hin, Win]

        # Convolution Layer 1 parameters
        Cout = 6
        cnv_kernel_size = 3
        cnv_kernel_size2 = 3
        cnv_stride2 = 1
        cnv_stride = 1
        cnv_padding = 1  # verify that it satisfies equation
        cnv_dilation = 1

        # Pooling Layer parameters
        pool_kernel_size = 2
        pool_stride = 2
        pool_padding = 0
        pool_dilation = 1

        # dense layer size from NN w/ BC
        L_Fout1 = 7000
        L_Fout2 = int(Hin*Win)

        # ---------------------------------------------------------------------
        # Layer 1 definition
        # ---------------------------------------------------------------------
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels= Cin,
                out_channels=Cout,
                kernel_size=cnv_kernel_size,
                stride=cnv_stride,
                padding=cnv_padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=pool_kernel_size,
                stride=pool_stride
            ),
            nn.Conv2d(
                in_channels=Cout,
                out_channels=Cout,
                kernel_size=cnv_kernel_size,
                stride=cnv_stride,
                padding=cnv_padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=pool_kernel_size,
                stride=pool_stride),
            nn.Flatten(),
            nn.LazyLinear(L_Fout1),
            nn.Linear(L_Fout1, L_Fout2)
        )

    def forward(self, x):
        return self.model(x).view((-1, *self.grid_size))

    def configure_optimizers(self, opt=torch.optim.Adam, **kwargs):
        optimizer = opt(self.model.parameters(), **kwargs)
        return optimizer

    def configure_loss(self, loss_fun=F.mse_loss):
        self.loss_fun = loss_fun

    def training_step(self, train_batch, train_batch_idx):
        x, y = train_batch
        y_hat = self.model(x).view((-1, *self.grid_size))
        loss = self.loss_fun(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        x, y = val_batch
        y_hat = self.model(x).view((-1, 1, *self.grid_size))
        loss = self.loss_fun(y_hat, y)
        self.log('val_loss', loss)
        return loss

    @property
    def shape(self):
        return self.grid_size

    @property
    def feature_names(self):
        return self.in_vars, self.out_vars
