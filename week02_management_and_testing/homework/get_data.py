import os
from torchvision.datasets import CIFAR10
from torchvision import transforms

def download_cifar10(data_dir="cifar10"):
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])

    CIFAR10(root=data_dir, train=True, download=True)
    CIFAR10(root=data_dir, train=False, download=True)

if __name__ == "__main__":
    download_cifar10()
