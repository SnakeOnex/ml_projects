from torchvision import transforms, datasets
from torch.utils.data import Subset
from vqvae import VQVAEConfig

mnist_stats = ([0.5], [0.5])
mnist_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mnist_stats[0], mnist_stats[1]),
])
mnist_vqvae_config = VQVAEConfig(in_channels=1, image_sz=28, ch_base=16, ch_mult=(1,2), K=32, D=8)
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
cifar10_vqvae_config = VQVAEConfig(in_channels=3, image_sz=32, ch_base=32, ch_mult=(1,2), K=512, D=64)
cifar10_config = {
    'vqvae_config': cifar10_vqvae_config,
    'stats': cifar10_stats,
    'fetch_train': lambda: datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar10_trans),
    'fetch_test': lambda: datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar10_trans),
}

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
imagenet_trans = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(48),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_stats[0], std=imagenet_stats[1]),
])
path_to_imagenet='/datagrid/public_datasets/imagenet/imagenet_pytorch'
train_sz, test_sz = 40_000, 2_000
imagenet_config = {
    'K': 512,
    'D': 64,
    'channels': 3,
    'image_sz': 128,
    'stats': imagenet_stats,
    'fetch_train': lambda: Subset(datasets.ImageNet(root=path_to_imagenet, split='train', transform=imagenet_trans), range(train_sz)),
    'fetch_test': lambda: Subset(datasets.ImageNet(root=path_to_imagenet, split='val', transform=imagenet_trans), range(test_sz)),
}

model_configs = {
    'mnist': mnist_config,
    'cifar10': cifar10_config,
    'imagenet': imagenet_config,
}
