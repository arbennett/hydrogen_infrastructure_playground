import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

class RMM_NN_2D_B1(pl.LightningModule):

    def __init__(self, grid_size=[25, 25],  channels=2, verbose=False,):
        super().__init__()
        self.use_dropout = False
        self.verbose = verbose

        # Inputs Parameters
        Cin = channels
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
            ).float(),
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
            ).float(),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=pool_kernel_size,
                stride=pool_stride),
            nn.LazyLinear(L_Fout1),
            nn.Linear(L_Fout1, L_Fout2)
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)
