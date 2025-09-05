import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from pathlib import Path
from modeling.diffusion import DiffusionModel


def train_step(model: DiffusionModel, inputs: torch.Tensor, optimizer: Optimizer, device: str):
    optimizer.zero_grad()
    inputs = inputs.to(device)
    loss = model(inputs)
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(model: DiffusionModel, dataloader: DataLoader, optimizer: Optimizer, device: str):
    model.train()
    pbar = tqdm(dataloader)
    loss_ema = None
    for x, _ in pbar:
        train_loss = train_step(model, x, optimizer, device)
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        pbar.set_description(f"loss: {loss_ema:.4f}")
    return loss_ema


def generate_samples(model: DiffusionModel, device: str, sample_path: str | Path, noize_path: str | Path | None = None):
    model.eval()
    with torch.no_grad():
        samples, init_noize = model.sample(8, (3, 32, 32), device=device)
        grid = make_grid(samples, nrow=4)
        save_image(grid, sample_path)
        if noize_path is not None:
            noize_grid = make_grid(init_noize, nrow=4)
            save_image(noize_grid, noize_path)

    return samples, init_noize


@torch.no_grad()
def full_noise_reconstruction(diffusion_model, dataloader, device, num_images=8,
                              original_path: str | Path | None = None, noised_path: str | Path | None = None,
                              reconstructed_path: str | Path | None = None):
    # images: батч из даталодера, нормализация как в обучении
    diffusion_model.eval()
    images, _ = next(iter(dataloader))
    x0 = torch.Tensor(images[:num_images]).to(device)

    # Полностью зашумляем из x0 (t = T)
    t_full = torch.tensor([diffusion_model.num_timesteps], device=device, dtype=torch.long)
    t_full = t_full.expand(x0.size(0))  # (B,)

    x_T = diffusion_model.q_sample(x0, t_full)  # из исходных картинок
    x_hat = diffusion_model.denoise_from(x_T, t_full[0])  # восстановление из x_T

    # Создаём гриды и сохраняем их
    if original_path is not None:
        original_grid = make_grid(x0, nrow=4, normalize=True, value_range=(-1, 1))
        save_image(original_grid, original_path)
    if noised_path is not None:
        fully_noised_grid = make_grid(x_T, nrow=4, normalize=True)
        save_image(fully_noised_grid, noised_path)
    if reconstructed_path:
        reconstructed_grid = make_grid(x_hat, nrow=4, normalize=True, value_range=(-1, 1))
        save_image(reconstructed_grid, reconstructed_path)

    return x0, x_T, x_hat
