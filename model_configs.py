from torchvision import transforms, datasets
from torch.utils.data import Subset


mnist_trans = transforms.Compose([
    transforms.ToTensor(),
])
mnist_config = {
    'K': 32,
    'D': 8,
    'channels': 1,
    'fetch_train': lambda: datasets.MNIST(root='./data', train=True, download=True, transform=mnist_trans),
    'fetch_test': lambda: datasets.MNIST(root='./data', train=False, download=True, transform=mnist_trans),
}

cifar10_trans = transforms.Compose([
    transforms.ToTensor(),
])
cifar10_config = {
    'K': 512,
    'D': 64,
    'channels': 3,
    'fetch_train': lambda: datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar10_trans),
    'fetch_test': lambda: datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar10_trans),
}

imagenet_trans = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
]),
path_to_imagenet='/datagrid/public_datasets/imagenet/imagenet_pytorch'
train_sz, test_sz = 10_000, 1_000
imagenet_config = {
    'K': 512,
    'D': 64,
    'channels': 3,
    'fetch_train': lambda: Subset(datasets.ImageNet(root=path_to_imagenet, split='train', transform=imagenet_trans), range(train_sz)),
    'fetch_test': lambda: Subset(datasets.ImageNet(root=path_to_imagenet, split='val', transform=imagenet_trans), range(test_sz))
}

model_configs = {
    'mnist': mnist_config,
    'cifar10': cifar10_config,
    'imagenet': imagenet_config,
}
