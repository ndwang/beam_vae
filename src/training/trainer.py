"""VAE Trainer class for managing training loops."""

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import vae_loss

if TYPE_CHECKING:
    from src.utils.logging import LoggingCallback


class Trainer:
    """Trainer class for VAE models.

    Args:
        model: VAE model to train.
        optimizer: PyTorch optimizer.
        scheduler: Optional learning rate scheduler.
        device: Device to train on.
        beta: KL divergence weight for beta-VAE.
        loss_type: Type of reconstruction loss ('mse' or 'bce').
        grad_clip: Maximum gradient norm for clipping.
        logger_callback: Optional logging callback for metrics and artifacts.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = None,
        beta: float = 0.0,
        loss_type: str = "mse",
        grad_clip: float = 1.0,
        logger_callback: Optional["LoggingCallback"] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = beta
        self.loss_type = loss_type
        self.grad_clip = grad_clip

        # Logging callback (import here to avoid circular imports)
        if logger_callback is None:
            from src.utils.logging import NoOpCallback
            logger_callback = NoOpCallback()
        self.logger_callback = logger_callback

        # Best model tracking for checkpointing
        self.best_val_loss = float('inf')

        self.model.to(self.device)

        self.history = {
            "train_total": [], "train_recon": [], "train_kl": [],
            "val_total": [], "val_recon": [], "val_kl": []
        }

        # Track starting epoch for resume functionality
        self.start_epoch = 0

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """Load a checkpoint to resume training.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            The epoch number to resume from.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state if available
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load beta if it was saved (for consistency check)
        saved_beta = checkpoint.get("beta")
        if saved_beta is not None and saved_beta != self.beta:
            print(f"Warning: Checkpoint beta={saved_beta} differs from current beta={self.beta}")

        # Set best validation loss from checkpoint
        self.best_val_loss = checkpoint.get("val_loss", float('inf'))

        # Return the epoch to resume from
        resume_epoch = checkpoint.get("epoch", 0)
        self.start_epoch = resume_epoch
        print(f"Resuming from epoch {resume_epoch} with val_loss={self.best_val_loss:.6f}")

        return resume_epoch

    def train_epoch(self, train_loader: DataLoader, max_steps: Optional[int] = None) -> Dict[str, float]:
        """Run one training epoch.

        Args:
            train_loader: DataLoader for training data.
            max_steps: Optional maximum number of steps per epoch.

        Returns:
            Dictionary with average losses for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_samples = 0

        loop = tqdm(train_loader, desc="Training", leave=False)
        for step, x in enumerate(loop):
            if max_steps is not None and step >= max_steps:
                break

            x = x.to(self.device)
            self.optimizer.zero_grad()

            recon, mu, logvar = self.model(x)
            loss, recon_loss, kl_loss = vae_loss(
                recon, x, mu, logvar, self.beta, self.loss_type
            )

            if torch.isnan(loss):
                raise ValueError(f"NaN loss detected at step {step}")

            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_recon += recon_loss.item() * batch_size
            total_kl += kl_loss.item() * batch_size
            n_samples += batch_size

        return {
            "total": total_loss / n_samples,
            "recon": total_recon / n_samples,
            "kl": total_kl / n_samples,
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            Dictionary with average losses.
        """
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_samples = 0

        for x in val_loader:
            x = x.to(self.device)
            recon, mu, logvar = self.model(x)
            loss, recon_loss, kl_loss = vae_loss(
                recon, x, mu, logvar, self.beta, self.loss_type
            )

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_recon += recon_loss.item() * batch_size
            total_kl += kl_loss.item() * batch_size
            n_samples += batch_size

        return {
            "total": total_loss / n_samples,
            "recon": total_recon / n_samples,
            "kl": total_kl / n_samples,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        max_steps: Optional[int] = None,
        save_dir: Optional[Path] = None,
        model_name: str = "vae",
        checkpoint_freq: int = 50,
    ) -> Dict[str, list]:
        """Train the model for multiple epochs.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Total number of epochs to train (not additional epochs).
            max_steps: Optional maximum steps per epoch.
            save_dir: Directory to save model and history.
            model_name: Base name for saved files.
            checkpoint_freq: Save checkpoint every N epochs.

        Returns:
            Training history dictionary.
        """
        for epoch in range(self.start_epoch, epochs):
            train_metrics = self.train_epoch(train_loader, max_steps)
            val_metrics = self.validate(val_loader)

            # Update history
            self.history["train_total"].append(train_metrics["total"])
            self.history["train_recon"].append(train_metrics["recon"])
            self.history["train_kl"].append(train_metrics["kl"])
            self.history["val_total"].append(val_metrics["total"])
            self.history["val_recon"].append(val_metrics["recon"])
            self.history["val_kl"].append(val_metrics["kl"])

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["total"])
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train: {train_metrics['total']:.6f} | "
                f"Val: {val_metrics['total']:.6f} | "
                f"LR: {current_lr:.2e}"
            )

            # Log metrics to callback
            self.logger_callback.log_metrics({
                "train/total_loss": train_metrics["total"],
                "train/recon_loss": train_metrics["recon"],
                "train/kl_loss": train_metrics["kl"],
                "val/total_loss": val_metrics["total"],
                "val/recon_loss": val_metrics["recon"],
                "val/kl_loss": val_metrics["kl"],
                "learning_rate": current_lr,
            }, step=epoch + 1)

            # Checkpointing
            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

                # Save best model
                if val_metrics["total"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["total"]
                    best_path = save_dir / f"{model_name}_best.pth"
                    self._save_checkpoint(best_path, epoch + 1, train_metrics, val_metrics)
                    self.logger_callback.log_checkpoint_metadata(
                        best_path, epoch + 1, val_metrics["total"], is_best=True
                    )

                # Periodic checkpoint
                if (epoch + 1) % checkpoint_freq == 0:
                    ckpt_path = save_dir / f"{model_name}_epoch{epoch + 1}.pth"
                    self._save_checkpoint(ckpt_path, epoch + 1, train_metrics, val_metrics)
                    self.logger_callback.log_checkpoint_metadata(
                        ckpt_path, epoch + 1, val_metrics["total"], is_best=False
                    )

        # Save final model and history
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            model_path = save_dir / f"{model_name}.pth"
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved to: {model_path}")

            history_path = save_dir / f"{model_name}_history.csv"
            self._save_history(history_path, epochs)
            print(f"History saved to: {history_path}")

        return self.history

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        """Save a full checkpoint with model, optimizer, and scheduler state.

        Args:
            path: Path to save the checkpoint.
            epoch: Current epoch number.
            train_metrics: Training metrics (total, recon, kl).
            val_metrics: Validation metrics (total, recon, kl).
        """
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "train_loss": train_metrics["total"],
            "train_recon_loss": train_metrics["recon"],
            "train_kl_loss": train_metrics["kl"],
            "val_loss": val_metrics["total"],
            "val_recon_loss": val_metrics["recon"],
            "val_kl_loss": val_metrics["kl"],
            "beta": self.beta,
        }, path)
        print(f"Checkpoint saved: {path}")

    def _save_history(self, path: Path, epochs: int) -> None:
        """Save training history to CSV file."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_total", "train_recon", "train_kl",
                "val_total", "val_recon", "val_kl"
            ])
            # History only contains entries for epochs we actually ran
            num_recorded = len(self.history["train_total"])
            for i in range(num_recorded):
                # Epoch number accounts for any resumed training
                epoch_num = self.start_epoch + i + 1
                writer.writerow([
                    epoch_num,
                    self.history["train_total"][i],
                    self.history["train_recon"][i],
                    self.history["train_kl"][i],
                    self.history["val_total"][i],
                    self.history["val_recon"][i],
                    self.history["val_kl"][i],
                ])
