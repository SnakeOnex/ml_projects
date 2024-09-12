import torch
from pathlib import Path
from torchvision import transforms, datasets
from torchvision.transforms import v2
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset

def make_loader(dataset, bs, shuffle, num_workers=2):
    return DataLoader(
            dataset, 
            batch_size=bs, 
            shuffle=shuffle, 
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True)

image_transforms = transforms.Compose([
    v2.Resize(256),
    v2.CenterCrop(256),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

pil_transforms = transforms.Compose([
    v2.Resize(256),
    v2.CenterCrop(256),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

class BirdDataset(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.data = []
        for y, subdir in enumerate(Path(path).iterdir()):
            if subdir.is_dir():
                for img_pth in subdir.iterdir():
                    self.data.append((str(img_pth), y))

    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        img = read_image(self.data[idx][0])
        label = torch.tensor(self.data[idx][1])
        if self.transform is not None: img = self.transform(img)
        return img, label

class FlowerDataset(Dataset):
    def __init__(self, path, transform=None):
        self.data = [(Path(p), None) for p in Path(path).iterdir()]
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img = read_image(str(self.data[idx][0]))
        if self.transform is not None: img = self.transform(img)
        return img, torch.tensor(0)

def make_bird_loader(bs):
    bird_dataset = BirdDataset(path='../../train', transform=image_transforms)
    train_set_sz = int(len(bird_dataset)*0.9)
    test_set_sz = len(bird_dataset) - train_set_sz
    train_bird_set, test_bird_set = torch.utils.data.random_split(bird_dataset, [train_set_sz, test_set_sz])
    return make_loader(train_bird_set, bs=bs, shuffle=True), make_loader(test_bird_set, bs=bs, shuffle=False)

def make_flower_loader(bs):
    flower_dataset = FlowerDataset(path='flowers', transform=image_transforms)
    train_set_sz = int(len(flower_dataset)*0.9)
    test_set_sz = len(flower_dataset) - train_set_sz
    train_flower_set, test_flower_set = torch.utils.data.random_split(flower_dataset, [train_set_sz, test_set_sz])
    return make_loader(train_flower_set, bs=bs, shuffle=True), make_loader(test_flower_set, bs=bs, shuffle=False)

def make_imagenet_loader(bs):
    path_to_imagenet='/mnt/data/Public_datasets/imagenet/imagenet_pytorch'
    train_sz, test_sz = 200_000, 20_000
    train_set = datasets.ImageNet(root=path_to_imagenet, split='train', transform=pil_transforms)
    test_set = datasets.ImageNet(root=path_to_imagenet, split='val', transform=pil_transforms)
    train_set, _ = torch.utils.data.random_split(train_set, [train_sz, len(train_set)-train_sz])
    test_set, _ = torch.utils.data.random_split(test_set, [test_sz, len(test_set)-test_sz])
    return make_loader(train_set, bs=bs, shuffle=True), make_loader(test_set, bs=bs, shuffle=False)

dataset_loaders = {
    "bird": make_bird_loader,
    "flower": make_flower_loader,
    "imagenet": make_imagenet_loader,
}
