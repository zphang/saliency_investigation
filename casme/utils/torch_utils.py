from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS

import pyutils.io as io


class ImagePathDataset(VisionDataset):
    def __init__(self, config, transform=None, target_transform=None,
                 loader=default_loader, return_paths=False):
        super().__init__(root=config["root"], transform=transform, target_transform=target_transform)
        self.config = config

        self.loader = loader
        self.extensions = IMG_EXTENSIONS

        self.classes = config["classes"]
        self.class_to_idx = config["class_to_idx"]
        self.samples = config["samples"]
        self.targets = [s[1] for s in self.samples]
        self.return_paths = return_paths

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        output = sample, target

        if self.return_paths:
            return output, path
        else:
            return output

    def __len__(self):
        return len(self.samples)

    @classmethod
    def from_path(cls, config_path, *args, **kwargs):
        return cls(config=io.read_json(config_path), *args, **kwargs)


def set_requires_grad(named_parameters, requires_grad):
    for name, param in named_parameters:
        param.requires_grad = requires_grad
