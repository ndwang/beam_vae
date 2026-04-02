"""Configuration validation using Pydantic models."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    """Validation schema for model configuration."""

    model_config = {"extra": "forbid"}

    name: Literal["vae2d", "residual_vae2d"] = "vae2d"
    input_channels: int = Field(default=15, ge=1)
    hidden_channels: List[int] = Field(default=[32, 64, 128, 256, 512])
    latent_dim: int = Field(default=64, ge=1)
    input_size: int = Field(default=64, ge=1)
    kernel_size: int = Field(default=3, ge=1)
    activation: str = "relu"
    output_activation: str = "sigmoid"
    batch_norm: bool = True
    dropout_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    weight_init: Literal["kaiming_normal", "xavier_normal", "xavier_uniform"] = "kaiming_normal"
    use_reparameterization: bool = True
    n_scales: int = Field(default=6, ge=0)
    n_centroids: int = Field(default=6, ge=0)

    @field_validator("hidden_channels")
    @classmethod
    def check_hidden_channels(cls, v: List[int]) -> List[int]:
        if len(v) < 1:
            raise ValueError("hidden_channels must have at least one element")
        if any(c <= 0 for c in v):
            raise ValueError("All hidden_channels must be positive")
        return v

    @model_validator(mode="after")
    def check_input_divisibility(self) -> "ModelConfig":
        n_downsamples = len(self.hidden_channels)
        divisor = 2 ** n_downsamples
        if self.input_size % divisor != 0:
            raise ValueError(
                f"input_size={self.input_size} must be divisible by 2^{n_downsamples}={divisor} "
                f"(number of encoder blocks)"
            )
        return self


class SchedulerConfig(BaseModel):
    """Validation schema for learning rate scheduler configuration."""

    model_config = {"extra": "forbid"}

    name: Literal["reduce_on_plateau", "cosine", "none"] = "reduce_on_plateau"
    factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    patience: int = Field(default=10, ge=1)


class WandbConfig(BaseModel):
    """Validation schema for Weights & Biases configuration."""

    model_config = {"extra": "forbid"}

    enabled: bool = False
    project: str = "beam-vae"
    entity: Optional[str] = None
    group: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    offline: bool = True


class TrainingConfig(BaseModel):
    """Validation schema for training configuration."""

    model_config = {"extra": "forbid"}

    epochs: int = Field(default=500, ge=1)
    batch_size: int = Field(default=256, ge=1)
    lr: float = Field(default=5e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    beta: float = Field(default=0.0, ge=0.0)
    gamma: float = Field(default=1.0, ge=0.0)
    delta: float = Field(default=0.0, ge=0.0)
    loss_type: Literal["mse", "weighted_mse", "bce"] = "mse"
    loss_config: dict = Field(default_factory=dict)
    grad_clip: float = Field(default=1.0, ge=0.0)
    val_split: float = Field(default=0.1, gt=0.0, lt=1.0)
    seed: int = 42
    num_workers: int = Field(default=8, ge=0)
    checkpoint_freq: int = Field(default=50, ge=1)
    max_steps: Optional[int] = Field(default=None, ge=1)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)


class DataConfig(BaseModel):
    """Validation schema for data configuration."""

    model_config = {"extra": "forbid"}

    name: Optional[str] = None
    path: str
    scales_path: str
    centroids_path: Optional[str] = None
    scaler_path: Optional[str] = None
    channels: int = Field(default=15, ge=1)
    height: int = Field(default=64, ge=1)
    width: int = Field(default=64, ge=1)

    @field_validator("path")
    @classmethod
    def check_path_format(cls, v: str) -> str:
        if not v:
            raise ValueError("Data path cannot be empty")
        return v


class Config(BaseModel):
    """Top-level configuration validation schema."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig
    run_name: Optional[str] = None
    output_dir: str = "./runs"

    model_config = {"extra": "forbid"}  # Reject unknown fields


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: List[Dict[str, Any]]):
        self.errors = errors
        messages = []
        for err in errors:
            loc = ".".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "Unknown error")
            messages.append(f"  {loc}: {msg}")
        super().__init__("Configuration validation failed:\n" + "\n".join(messages))


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration dictionary against schema.

    Args:
        config: Raw configuration dictionary.

    Returns:
        Validated and normalized configuration dictionary.

    Raises:
        ConfigValidationError: If validation fails with details about the errors.
    """
    try:
        validated = Config(**config)
        return validated.model_dump()
    except Exception as e:
        if hasattr(e, "errors"):
            raise ConfigValidationError(e.errors())
        raise ConfigValidationError([{"loc": [], "msg": str(e)}])
