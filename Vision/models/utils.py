import random
import os
import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def calc_accuracy(pred, label):
    _, max_indices = torch.max(pred, 1)
    accuracy = (max_indices == label).sum().data.cpu().numpy() / max_indices.size()[0]

    pred = torch.argmax(pred, dim=-1).data.cpu().numpy()
    label = label.data.cpu().numpy()

    return accuracy
