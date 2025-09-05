from pydantic import BaseModel, Field
from typing import Tuple, Literal


class ModelConfig(BaseModel):
    hidden_size: int = Field(..., ge=32, le=1024)
    betas: Tuple[float, float]
    num_timesteps: int = Field(..., gt=0)


class TrainerConfig(BaseModel):
    num_epochs: int = Field(..., gt=0)
    batch_size: int = Field(..., gt=0)
    lr: float = Field(..., gt=0)
    device: Literal["cpu", "cuda", "mps"]
    num_dataloader_workers: int = Field(..., gt=0)
    optimizer: dict

class WandbConfig(BaseModel):
    project: str
    run_name: str | None = None
    api_key: str


class Config(BaseModel):
    model: ModelConfig
    train: TrainerConfig
    wandb: WandbConfig
