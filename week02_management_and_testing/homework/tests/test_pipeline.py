import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import train_step, train_epoch, generate_samples
from modeling.unet import UnetModel


torch.manual_seed(2674)


@pytest.fixture
def train_dataset():
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(
        "../cifar10",
        train=True,
        download=True,
        transform=transforms,
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["mps"]])
def test_train_on_one_batch(device, train_dataset):
    # note: you should not need to increase the threshold or change the hyperparameters
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    x, _ = next(iter(dataloader))
    loss = None
    for i in range(50):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5


@pytest.mark.parametrize(["device"], [["cpu"], ["mps"]])
def test_training(device, train_dataset):
    # note: implement and test a complete training procedure (including sampling)
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=128),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    if device == "cpu":
        cropped_dataset_len = int(0.1 * len(train_dataset))
        dataset = Subset(train_dataset, indices=range(cropped_dataset_len))
    else:
        dataset = train_dataset

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optim = torch.optim.Adam(ddpm.parameters(), lr=3e-4)

    samples_dir = Path("samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    loss_ema = train_epoch(ddpm, dataloader, optim, device)
    assert loss_ema < 0.1

    samples, _ = generate_samples(ddpm, device, samples_dir / f"{0:02d}.png")
    assert samples.shape == (8, 3, 32, 32)
    assert not torch.any(torch.isnan(samples)), "No NaNs in samples"
    assert not torch.any(torch.isinf(samples)), "No Infs in samples"
