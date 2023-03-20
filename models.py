# Import pytorch modules 

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import pytorch lightning modules 

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Import other modules

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import glob
# from PIL import Image
from torchvision.models import resnet18
import matplotlib.pyplot as plt


class Encoder(torch.nn.Module): 
    def __init__(self): 
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()
        self.resnet.eval()

    def forward(self, x):
        return self.resnet(x).reshape(-1, 512, 8, 8)
    
class Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv6 = nn.Conv2d(16, 3, 3, padding=1)
        self.conv7 = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        '''
            Output shape: (batch_size, 3, 256, 256)
        '''
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv3(x))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv4(x))
        # x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv5(x))
        # x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv6(x))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv7(x))
        return x


class SSLModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.corrupt(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def corrupt(self, x): 
        '''Corrupt the image by choosing random patches and interchanging their values.

        Args:
            x: (3, 256, 256)

        Returns:
            x: (3, 256, 256)S
        '''

        for i in range(10):
            x = self.swap_patches(x)

        return x

    def swap_patches(self, x):
        '''Swap two random patches of size 32x32.

        Args:
            x: (3, 256, 256)

        Returns:
            x: (3, 256, 256)
        '''

        x1, y1 = torch.randint(0, 256-32, (2,))
        x2, y2 = torch.randint(0, 256-32, (2,))
        x[:, x1:x1+32, y1:y1+32], x[:, x2:x2+32, y2:y2+32] = x[:, x2:x2+32, y2:y2+32], x[:, x1:x1+32, y1:y1+32]

        return x