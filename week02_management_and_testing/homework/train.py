import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid

import wandb
import hydra
from hydra_slayer import get_from_params
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

from pathlib import Path

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch, full_noise_reconstruction
from modeling.unet import UnetModel

from utils.config_validation import Config


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig):
    validated_config = Config(**cfg)
    model_config, train_config, wandb_config = validated_config.model, validated_config.train, validated_config.wandb

    wandb.login(key=wandb_config.api_key)
    with wandb.init(
        project=wandb_config.project,
        name=wandb_config.run_name,
        config=OmegaConf.to_container(cfg),
    ) as run:

        ddpm = DiffusionModel(
            eps_model=UnetModel(3, 3, hidden_size=model_config.hidden_size),
            betas=model_config.betas,
            num_timesteps=model_config.num_timesteps,
        )
        ddpm.to(train_config.device)
        run.watch(ddpm)

        train_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        train_dataset = CIFAR10(
            "cifar10",
            train=True,
            download=True,
            transform=train_transforms,
        )
        test_dataset = CIFAR10(
            "cifar10",
            train=False,
            download=True,
            transform=train_transforms,
        )

        train_dataloader = DataLoader(train_dataset, batch_size=train_config.batch_size, num_workers=train_config.num_dataloader_workers, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=train_config.batch_size, num_workers=train_config.num_dataloader_workers, shuffle=False)

        optim = get_from_params(**train_config.optimizer)
        optim = optim(ddpm.parameters(), lr=train_config.lr)

        samples_dir = Path("samples")
        samples_dir.mkdir(parents=True, exist_ok=True)
        reconstruction_dir = Path("reconstruction")
        reconstruction_dir.mkdir(parents=True, exist_ok=True)

        for i in range(train_config.num_epochs):
            loss_ema = train_epoch(ddpm, train_dataloader, optim, train_config.device)
            samples, init_noize = generate_samples(ddpm, train_config.device, samples_dir / f"{i:02d}_samples.png",
                                                   samples_dir / f"{i:02d}_init_noize.png")
            originals, noised, reconstructed = full_noise_reconstruction(ddpm, test_dataloader, train_config.device, original_path=reconstruction_dir / f"{i:02d}_original.png",
                                      noised_path=reconstruction_dir / f"{i:02d}_noised.png", reconstructed_path=reconstruction_dir / f"{i:02d}_reconstructed.png")

            curr_stats = {
                "loss/curr_epoch_ema_loss": loss_ema,
                "samples/init_noise": wandb.Image(make_grid(init_noize, nrow=4, normalize=True, value_range=(-1, 1))),
                "samples/generated_pic": wandb.Image(make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))),
                "reconstruction/original_image": wandb.Image(make_grid(originals, nrow=4, normalize=True, value_range=(-1, 1))),
                "reconstruction/noised_image": wandb.Image(make_grid(noised, nrow=4, normalize=True, value_range=(-1, 1))),
                "reconstruction/reconstructed_image": wandb.Image(make_grid(reconstructed, nrow=4, normalize=True, value_range=(-1, 1))),

            }
            run.log(curr_stats)


if __name__ == "__main__":
    load_dotenv()
    main()
