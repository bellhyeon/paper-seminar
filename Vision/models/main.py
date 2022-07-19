import argparse
import torch
from torchvision import transforms
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchsummary import summary
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from utils import seed_everything, calc_accuracy
from model.modeling import Model

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--num_features", type=int, default=100)
    args.add_argument("--random_state", type=int, default=42)
    args.add_argument("--batch_size", type=int, default=128)
    args.add_argument("--num_workers", type=int, default=4)
    args.add_argument("--learning_rate", type=float, default=1e-4)
    args.add_argument("--weight_decay", type=float, default=0.0)
    args.add_argument("--epochs", type=int, default=90)
    args.add_argument("--model_name", type=str, default="googlenet")
    args.add_argument("--img_size", type=int, default=224)
    args = args.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.random_state)

    # ===========================================================================
    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
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
        train_data, test_size=0.2, random_state=args.random_state, shuffle=True
    )

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # ===========================================================================
    model = Model.get_model(args.model_name, args.__dict__).to(device)
    model.init_weights()
    summary(model, (3, 227, 227))

    # ===========================================================================
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    # optimizer = SGD(
    #     model.parameters(),
    #     lr=CFG.lr,
    #     momentum=CFG.momentum,
    #     weight_decay=CFG.weight_decay,
    # )
    optimizer = Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # ===========================================================================
    prev_val_loss = 1e4
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        total_train_loss, total_val_loss = 0.0, 0.0
        total_train_acc, total_val_acc = 0.0, 0.0
        model.train()

        for img, label in tqdm(train_loader):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()

            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_acc += calc_accuracy(output, label)
        model.eval()
        with torch.no_grad():
            for img, label in tqdm(val_loader):
                img, label = img.to(device), label.to(device)
                output = model(img)
                loss = criterion(output, label)
                total_val_loss += loss.item()
                total_val_acc += calc_accuracy(output, label)

        train_loss = round(total_train_loss / len(train_loader), 5)
        train_acc = round(total_train_acc / len(train_loader), 4)
        val_loss = round(total_val_loss / len(val_loader), 5)
        val_acc = round(total_val_acc / len(val_loader), 4)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if prev_val_loss > val_loss:
            prev_val_loss = val_loss
        # if prev_val_loss < val_loss:
        #     CFG.lr /= 10.0
        print(
            f"Epoch {epoch+1}     train loss: {train_loss}     train acc: {train_acc}     valid loss: {val_loss}     val acc: {val_acc}     lr: {optimizer.param_groups[0]['lr']}"
        )
