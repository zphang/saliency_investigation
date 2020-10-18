import numpy as np

import torch

import torchvision.transforms as transforms
from casme.utils.torch_utils import ImagePathDataset

NORMALIZATION_MEAN = np.array([0.485, 0.456, 0.406])
NORMALIZATION_STD = np.array([0.229, 0.224, 0.225])
NORMALIZATION = transforms.Normalize(
    mean=NORMALIZATION_MEAN,
    std=NORMALIZATION_STD,
)
DENORMALIZATION = transforms.Normalize(
    mean=-NORMALIZATION_MEAN / NORMALIZATION_STD,
    std=1 / NORMALIZATION_STD,
)

NORMALIZATION_MEAN_TENSOR = torch.tensor(NORMALIZATION_MEAN).view(1, 3, 1, 1).float()
NORMALIZATION_STD_TENSOR = torch.tensor(NORMALIZATION_STD).view(1, 3, 1, 1).float()


def normalize_arr(x):
    # Works for both B,H,W,C as well as H,W,C
    return (x - NORMALIZATION_MEAN) / NORMALIZATION_STD


def denormalize_arr(x):
    # Works for both B,H,W,C as well as H,W,C
    return (x * NORMALIZATION_STD) + NORMALIZATION_MEAN


def normalize_tensor(x):
    return (x - NORMALIZATION_MEAN_TENSOR.to(x.device)) / NORMALIZATION_STD_TENSOR.to(x.device)


def denormalize_tensor(x):
    return (x * NORMALIZATION_STD_TENSOR.to(x.device)) + NORMALIZATION_MEAN_TENSOR.to(x.device)


def tensor_to_image_arr(x):
    if len(x.shape) == 4:
        return x.permute([0, 2, 3, 1]).cpu().numpy()
    elif len(x.shape) == 3:
        return x.permute([1, 2, 0]).cpu().numpy()
    else:
        raise RuntimeError(x.shape)


def get_data_loaders(train_json, val_json, batch_size, workers):
    if train_json:
        train_loader = torch.utils.data.DataLoader(
            ImagePathDataset.from_path(
                config_path=train_json,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    NORMALIZATION,
                ])),
            batch_size=batch_size, shuffle=True, num_workers=workers,
            pin_memory=False, sampler=None,
        )
    else:
        train_loader = None

    if val_json:
        val_loader = torch.utils.data.DataLoader(
            ImagePathDataset.from_path(
                config_path=val_json,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    NORMALIZATION,
                ])),
            batch_size=batch_size, shuffle=False, num_workers=workers,
            pin_memory=False,
        )

    else:
        val_loader = None

    return train_loader, val_loader
