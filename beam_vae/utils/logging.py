"""Logging callbacks for training metrics and model artifacts."""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class LoggingCallback(ABC):
    """Abstract base class for logging callbacks.

    Keeps the Trainer framework-agnostic by defining a protocol
    for logging metrics and model artifacts.
    """

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics at a given step.

        Args:
            metrics: Dictionary of metric names to values.
            step: Current step/epoch number.
        """
        pass

    @abstractmethod
    def finish(self) -> None:
        """Clean up and finalize logging."""
        pass


class NoOpCallback(LoggingCallback):
    """No-op callback when logging is disabled."""

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbCallback(LoggingCallback):
    """Weights & Biases logging callback.

    Args:
        run: W&B Run object.
    """

    def __init__(self, run):
        self.run = run

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to W&B."""
        self.run.log(metrics, step=step)

    def finish(self) -> None:
        """Finish the W&B run."""
        self.run.finish()
