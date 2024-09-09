import PIL, numpy as np, torch
from pathlib import Path
from torchvision import transforms, datasets
from torchvision.transforms import v2
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from vqgan import VQGANConfig
from vqvae import VQVAEConfig
from utils import compute_stats

mnist_stats = ([0.5], [0.5])
mnist_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mnist_stats[0], mnist_stats[1]),
])
mnist_vqvae_config = VQVAEConfig(in_channels=1, image_sz=28, ch_base=16, ch_mult=(1,2), K=512, D=64)
mnist_config = {
    'vqvae_config': mnist_vqvae_config,
    'stats': mnist_stats,
    'fetch_train': lambda: datasets.MNIST(root='./data', train=True, download=True, transform=mnist_trans),
    'fetch_test': lambda: datasets.MNIST(root='./data', train=False, download=True, transform=mnist_trans),
}


cifar10_stats = ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
cifar10_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_stats[0], cifar10_stats[1]),
])
cifar10_vqvae_config = VQVAEConfig(in_channels=3, image_sz=32, ch_base=64, ch_mult=(1,2,4), K=512, D=64)
cifar10_config = {
    'vqvae_config': cifar10_vqvae_config,
    'stats': cifar10_stats,
    'fetch_train': lambda: datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar10_trans),
    'fetch_test': lambda: datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar10_trans),
}

bird_stats = ([1.0, 1.0, 1.0], [127.5, 127.5, 127.5])
bird_trans = transforms.Compose([
    v2.Resize(256),
    v2.CenterCrop(256),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
bird_vqgan_config = VQGANConfig(K=2048, D=256, image_sz=256)

class BirdDataset(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.data = []
        for y, subdir in enumerate(Path(path).iterdir()):
            if subdir.is_dir():
                for img_pth in subdir.iterdir():
                    self.data.append((str(img_pth), y))
        print(f'Found {len(self.data)} images in {path}')

    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        img = read_image(self.data[idx][0])
        label = torch.tensor(self.data[idx][1])
        if self.transform is not None: img = self.transform(img)
        return img, label


bird_dataset = BirdDataset(path='../../train', transform=bird_trans)

train_set_sz = int(len(bird_dataset)*0.9)
test_set_sz = len(bird_dataset) - train_set_sz
train_bird_set, test_bird_set = torch.utils.data.random_split(bird_dataset, [train_set_sz, test_set_sz])

# use Subset
# train_bird_set = Subset(bird_dataset, range(0, 2000))
# test_bird_set = Subset(bird_dataset, range(2000, 2200))

bird_config = {
    'vqgan_config': bird_vqgan_config,
    'fetch_train': lambda: train_bird_set,
    'fetch_test': lambda: test_bird_set,
}

class FlowerDataset(Dataset):
    def __init__(self, path, transform=None):
        self.data = [(Path(p), None) for p in Path(path).iterdir()]
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img = read_image(str(self.data[idx][0]))
        if self.transform is not None: img = self.transform(img)
        return img, torch.tensor(0)

flower_dataset = FlowerDataset(path='flowers', transform=bird_trans)

train_set_sz = int(len(flower_dataset)*0.9)
test_set_sz = len(flower_dataset) - train_set_sz
train_flower_set, test_flower_set = torch.utils.data.random_split(flower_dataset, [train_set_sz, test_set_sz])

flower_config = {
    'vqgan_config': bird_vqgan_config,
    'fetch_train': lambda: train_flower_set,
    'fetch_test': lambda: test_flower_set,
}

# imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# imagenet_trans = transforms.Compose([
    # transforms.Resize(64),
    # transforms.CenterCrop(48),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=imagenet_stats[0], std=imagenet_stats[1]),
# ])
# path_to_imagenet='/datagrid/public_datasets/imagenet/imagenet_pytorch'

imagenet_trans = transforms.Compose([
    v2.Resize(256),
    v2.CenterCrop(256),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
path_to_imagenet='/mnt/data/Public_datasets/imagenet/imagenet_pytorch'
train_sz, test_sz = 100_000, 10_000
imagenet_config = {
    'vqgan_config': bird_vqgan_config,
    'fetch_train': lambda: Subset(datasets.ImageNet(root=path_to_imagenet, split='train', transform=imagenet_trans), range(train_sz)),
    'fetch_test': lambda: Subset(datasets.ImageNet(root=path_to_imagenet, split='val', transform=imagenet_trans), range(test_sz)),
}

model_configs = {
    'mnist': mnist_config,
    'cifar10': cifar10_config,
    'imagenet': imagenet_config,
    'bird': bird_config,
    'flower': flower_config
}

