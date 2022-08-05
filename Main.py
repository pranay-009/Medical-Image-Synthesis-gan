print("Running file Main5.py")

import os
import random

import numpy as np
import pandas as pd
import skimage
import skimage.io as io
import skimage.filters as skfilters

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image


class Dataset10x(Dataset):
    def __init__(self,
        csv_file,
        root_dir,
        transform=None,
        sample_count=500,
        image_size=None,
        clearance=20
    ) -> None:
        super().__init__()

        self.frames = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size if image_size and image_size > 0 else 128
        self.clearance = clearance
        self.sample_count = sample_count
        self.iter_count = 0

    def __len__(self) -> int:
        return self.sample_count

    def get_random_sample(self):
        return io.imread(
            os.path.join(
                self.root_dir,
                self.frames.iloc[random.randint(0, len(self.frames)-1), 0]
            )
        )

    def __getitem__(self, index) -> list:

        if self.iter_count >= self.sample_count:
            raise StopIteration

        if torch.is_tensor(index):
            index = index.tolist()

        img = self.get_random_sample()

        if self.image_size:
            x = random.randint(self.clearance, img.shape[0] - self.image_size - self.clearance)
            y = random.randint(self.clearance, img.shape[1] - self.image_size - self.clearance)
            img = img[x : x + self.image_size, y : y + self.image_size]

        img_blur = skfilters.gaussian(img)
        img_blur = img_blur.astype(np.float32)

        if self.transform:
            img = self.transform(img)
            img_blur = self.transform(img_blur)

        self.iter_count += 1
        return img, img_blur


class Generator(nn.Module):
    def __init__(self, z_dim, channels, features=64) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.channels = channels
        self.features = features

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.channels, self.features, 3, 4, 1),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = self._block(self.features, self.features * 4, 3, 4, 1)
        self.layer3 = self._block_rev(self.features * 4, self.features, 3, 1, 1)
        self.layer4 = self._block_rev(self.features, self.channels, 3, 1, 1)
        # self.layer5 = self._block_rev(self.features * 4, self.features, 3, 1, 1)
        self.layer5 = nn.Tanh()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def _block_rev(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=4),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding
            )
            # nn.ConvTranspose2d(
            #     out_channels,
            #     out_channels * 4,
            #     kernel_size,
            #     stride,
            #     padding
            # )
        )

    def forward(self, x):
        # x = 128 x 128 image
        # print(x.shape)
        x1 = self.layer1(x) # 32 x 32
        # print(x1.shape)
        x2 = self.layer2(x1) # 8 x 8
        # print(x2.shape)
        x3 = self.layer3(x2) # 32 x 32
        # print(x3.shape)
        x4 = self.layer4(x3) # 128 x 128
        # print(x4.shape)
        x5 = self.layer5(x4) # 128 x 128
        # print(x5.shape)
        # x6 = self.layer6(x5)
        # print(x6.shape)
        return x5

class Discriminator(nn.Module):
    def __init__(self, channels, features) -> None:
        super().__init__()
        self.channels = channels
        self.features = features
        self.model = nn.Sequential(
            nn.Conv2d(self.channels, self.features, 3, 4, 1), # 128 x 128
            nn.LeakyReLU(0.2),
            self._block(self.features, self.features * 4, 3, 4, 1), # 32 x 32
            self._block(self.features * 4, self.features * 8, 3, 4, 1), # 8 x 8
            nn.AvgPool2d(3, 4, 1),
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor):
        # x = x.type(torch.FloatTensor)
        return self.model(x)


# HyperParams
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
BATCH_SIZE = 5
IMAGE_SIZE = 128
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 100
FEATURES_DISC = 128
FEATURES_GEN = 128
TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        )
    ]
)

# Dataset and DataLoader
dataset = Dataset10x(r"./DATASET/files.csv", r"./DATASET", TRANSFORM, IMAGE_SIZE)
dataloader = DataLoader(dataset, 2, True)

# models
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(DEVICE)
dis = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(DEVICE)
gen_optim = optim.Adam(gen.parameters(), LEARNING_RATE, (0.5, 0.5))
dis_optim = optim.Adam(dis.parameters(), LEARNING_RATE, (0.5, 0.5))
# criterion = nn.SmoothL1Loss()
criterion = nn.BCELoss()

# train
gen.train()
dis.train()
for epoch in range(NUM_EPOCHS):
    for batch_idx, (img, blr_img) in enumerate(dataloader):
        real = img.to(DEVICE)
        blr = blr_img.to(DEVICE)
        fake = gen(blr)
        
        dis_real = dis(real).reshape(-1)
        dis_fake = dis(fake).reshape(-1)
        dis_loss = criterion(dis_real, torch.ones_like(dis_real)) + criterion(dis_fake, torch.zeros_like(dis_fake))
        dis.zero_grad()
        dis_loss.backward(retain_graph=True)
        dis_optim.step()


        # output = gen(blr)
        gen_loss = criterion(real, torch.ones_like(real)) + criterion(fake, torch.zeros_like(fake))
        gen.zero_grad()
        gen_loss.backward()
        gen_optim.step()


        if batch_idx == 0:
            print(
                f"EPOCH : [{epoch}/{NUM_EPOCHS}]\n\t"
                f"LOSS DIS : [{dis_loss:.4f}]\n\t"
                f"LOSS GEN : [{gen_loss:.4f}]\n"
            )

            with torch.no_grad():
                save_image(
                    fake,
                    f"GAN/fake_{epoch}.png"
                )