import torch
import torch.nn as nn
import torch.nn.functional as F

import loss

class RainNet(pl.LightningModule):
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = loss.LogCoshLoss()

        # Layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder layers
        self.down1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, bias=False, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
        )

        # Decoder layers
        self.upsample6 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        self.uconv6 = nn.Sequential(
            nn.Conv2d(1024 + 512, 512, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.upsample7 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.uconv7 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.upsample8 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.uconv8 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.upsample9 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.uconv9 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        conv1 = self.down1(x)
        pool1 = self.pool(conv1)

        conv2 = self.down2(pool1)
        pool2 = self.pool(conv2)

        conv3 = self.down3(pool2)
        pool3 = self.pool(conv3)

        conv4 = self.down4(pool3)
        pool4 = self.pool(conv4)

        conv5 = self.down5(pool4)

        up6 = self.upsample6(conv5)
        up6 = torch.cat([up6, conv4], dim=1)
        conv6 = self.uconv6(up6)

        up7 = self.upsample7(conv6)
        up7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.uconv7(up7)

        up8 = self.upsample8(conv7)
        up8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.uconv8(up8)

        up9 = self.upsample9(conv8)
        up9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.uconv9(up9)

        out = self.out(conv9)

        return out

    def training_step(self, batch, batch_idx):
        loss = self.shared_step()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val_loss", loss)

    def shared_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.paramters(), lr=self.hparams.lr)
        return optimizer