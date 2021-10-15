import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Union, List, Mapping

# Just for conceptual clarity
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class Conv3d_AE(pl.LightningModule):

    def __init__(
            self,
            grid_size=[5, 50, 50],
            in_vars: List[str]=[],
            out_vars: List[str]=[],
            embedding_dim: int=4096,
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
        Din, Hin, Win = grid_size
        flat_gridsize = np.prod(grid_size)
        in_shape = [Cin, Din, Hin, Win]
        self.activation = nn.Mish

        # Encoder definition
        # TODO: Should probalby implement some pooling
        self.encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=Cin,
                out_channels=Cin,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1
            ),
            self.activation(),
            nn.Conv3d(
                in_channels=Cin,
                out_channels=1,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=1
            ),
            self.activation(),
            nn.Flatten(),
            nn.Linear(flat_gridsize//4, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Linear(embedding_dim, flat_gridsize//4),
            Reshape(-1, 1, grid_size[0], *[g//2 for g in grid_size[1:]]),
            nn.ConvTranspose3d(
                in_channels=1,
                out_channels=Cin,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=1
            ),
            self.activation(),
            nn.ConvTranspose3d(
                in_channels=Cin,
                out_channels=Cin,
                kernel_size=(3, 4, 4),
                stride=(1, 2, 2),
                padding=1
            ),
            self.activation()
        )

        self.model = nn.Sequential(
            self.encoder,
            self.decoder
        )

    def forward(self, x):
        return self.encoder(x)

    def configure_optimizers(self, opt=torch.optim.Adam, **kwargs):
        optimizer = opt(self.model.parameters(), **kwargs)
        return optimizer

    def configure_loss(self, loss_fun=F.mse_loss):
        self.loss_fun = loss_fun

    def training_step(self, train_batch, train_batch_idx):
        x, y = train_batch
        y_hat = self.model(x)
        loss = self.loss_fun(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        x, y = val_batch
        y_hat = self.model(x)
        loss = self.loss_fun(y_hat, y)
        self.log('val_loss', loss)
        return loss

    @property
    def shape(self):
        return self.grid_size

    @property
    def feature_names(self):
        return self.in_vars, self.out_vars
