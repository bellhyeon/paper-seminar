import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision
import numpy as np
from torchsummary import summary
from sklearn.model_selection import train_test_split
from torch.optim import SGD, Adam
from tqdm import tqdm
import os
import random


class CFG:
    seed = 42
    batch_size = 128
    num_workers = 4
    lr = 1e-2
    momentum = 0.9
    weight_decay = 4e-5
    epochs = 90
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlexNet(nn.Module):
    """
    AlexNet Pytorch Implementation.
    It follows `ImageNet Classification with Deep Convolutional Neural Networks`.

    Structure: 5 convolutional layers + 3 fully-connected layers 

    """

    def __init__(self, num_features=1000):

        super(AlexNet, self).__init__()
        self.num_features = num_features
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=96, kernel_size=11, stride=4
            ),  # (bs, 3, 227, 227) → (bs, 96, 55, 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(
                kernel_size=3, stride=2
            ),  # (bs, 96, 55, 55) → (bs, 96, 27, 27)
            nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2
            ),  # (bs, 96, 27, 27) → (bs, 256, 27, 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(
                kernel_size=3, stride=2
            ),  # (bs, 256, 27, 27) → (bs, 256, 13, 13)
            nn.Conv2d(
                in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1
            ),  # (bs, 256, 13, 13) → (bs, 384, 13, 13)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1
            ),  # (bs, 384, 13, 13) → (bs, 384, 13, 13)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # (bs, 384, 13, 13), (bs, 256, 13, 13)
            nn.MaxPool2d(
                kernel_size=3, stride=2
            ),  # (bs, 256, 13, 13) → (bs, 256, 6, 6)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),  # same as x.view(-1, 256 * 6 * 6)
            nn.Dropout(p=0.5),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=self.num_features),
        )

    def init_weights(self):
        for layer_num, layer in enumerate(self.conv):
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01),
                nn.init.zeros_(layer.bias)
                if (
                    layer_num == 4 or layer_num == 10 or layer_num == 12
                ):  # second, fourth, fifth conv layers
                    nn.init.ones_(layer.bias)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01),
                nn.init.ones_(layer.bias)

    def forward(self, inp):
        conv_result = self.conv(inp)
        output = self.fc(conv_result)
        return output


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":

    seed_everything()

    transform = transforms.Compose(
        [
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_data = torchvision.datasets.CIFAR10(
        root="./cifar100", train=True, download=True, transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root="./cifar100", train=False, download=True, transform=transform
    )

    train_data, val_data = train_test_split(
        train_data, test_size=0.2, random_state=CFG.seed, shuffle=True
    )

    train_loader = DataLoader(
        train_data, batch_size=CFG.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=CFG.batch_size, shuffle=False, num_workers=0
    )

    model = AlexNet(num_features=100)
    model.init_weights()
    model = model.to(CFG.device)
    summary(model, (3, 227, 227))

    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=CFG.lr,
        momentum=CFG.momentum,
        weight_decay=CFG.weight_decay,
    )
    # optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    prev_val_loss = 1e4
    train_losses, val_losses = [], []

    for epoch in range(CFG.epochs):
        total_train_loss, total_val_loss = 0.0, 0.0

        model.train()

        for img, label in tqdm(train_loader):
            img, label = img.to(CFG.device), label.to(CFG.device)
            optimizer.zero_grad()

            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for img, label in tqdm(val_loader):
                img, label = img.to(CFG.device), label.to(CFG.device)
                output = model(img)
                loss = criterion(output, label)
                total_val_loss += loss.item()

        train_loss = round(total_train_loss / len(train_loader), 4)
        val_loss = round(total_val_loss / len(val_loader), 4)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if prev_val_loss > val_loss:
            prev_val_loss = val_loss
        if prev_val_loss < val_loss:
            CFG.lr /= 10.0
        print(
            f"Epoch {epoch+1}     train loss {train_loss}     valid loss {val_loss}     lr {CFG.lr}"
        )
