import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from VAE2D import VAE2D
from ResidualVAE2D import ResidualVAE2D
from FrequencyMapDataset import FrequencyMapDataset
import csv

# ============================================================
# === Training Function =======================================
# ============================================================

def train_vae2d(
    config,
    dataset_path="/pscratch/sd/n/ndwang/frequency_maps/frequency_maps_log_minmax.npy",
    batch_size=512,
    num_workers=8,
    epochs=300,
    lr=1e-3,
    weight_decay=1e-4,
    max_steps=None,
    B=0,
    return_history=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------------------------
    # Dataset + Split
    # --------------------------
    full_dataset = FrequencyMapDataset(dataset_path)
    N = len(full_dataset)
    val_size = int(0.1 * N)
    train_size = N - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # --------------------------
    # Create VAE2D model
    # --------------------------
    model = VAE2D(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10
    )

    MSEloss = nn.MSELoss()

    # Loss history
    history = {
        "train_total": [], "train_recon": [], "train_kl": [],
        "val_total": [],   "val_recon": [],   "val_kl": []
    }
    torch.autograd.set_detect_anomaly(True)
    # --------------------------
    # Training loop
    # --------------------------
    for epoch in range(epochs):
        model.train()
        train_total = 0
        train_recon = 0
        train_kl = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        global_step = 0

        for x in loop:
            x = x.to(device)
            optimizer.zero_grad()

            recon, mu, logvar = model(x)
            recon_loss = MSEloss(recon, x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - torch.exp(logvar))

            if torch.isnan(recon_loss):
                print(f"recon_loss is nan at: {global_step}")
                return
            if torch.isnan(kl_loss):
                print(f"kl_loss is nan at: {global_step}")
                return

            loss = recon_loss + B * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_total += loss.item() * x.size(0)
            train_recon += recon_loss.item() * x.size(0)
            train_kl += kl_loss.item() * x.size(0)

            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                print(f"Reached max_steps={max_steps}. Stopping early.")
                break

        # Normalize
        train_total /= train_size
        train_recon /= train_size
        train_kl /= train_size

        # --------------------------
        # Validation
        # --------------------------
        model.eval()
        val_total = 0
        val_recon = 0
        val_kl = 0

        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)

                recon_loss = MSEloss(recon, x)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - torch.exp(logvar))
                loss = recon_loss + B * kl_loss

                val_total += loss.item() * x.size(0)
                val_recon += recon_loss.item() * x.size(0)
                val_kl += kl_loss.item() * x.size(0)

        val_total /= val_size
        val_recon /= val_size
        val_kl /= val_size

        # Print metrics
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Total: {train_total} | Val Total: {val_total}"
        )

        # Save
        history["train_total"].append(train_total)
        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)

        history["val_total"].append(val_total)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)

        # Scheduler uses validation loss only
        scheduler.step(val_total)
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6e}")


    # --------------------------
    # Save model
    # --------------------------
    out = f"VAE2D_e{epochs}_B{B}_lr{lr}_latent{config['model']['latent_dim']}.pth"
    torch.save(model.state_dict(), out)
    print(f"Saved model to: {out}")

    # --------------------------
    # Save CSV
    # --------------------------
    csv_filename = f"loss_history_VAE2D_e{epochs}_B{B}_lr{lr}_latent{config['model']['latent_dim']}_MSE.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_total", "train_recon", "train_kl",
            "val_total", "val_recon", "val_kl"
        ])
        for ep in range(epochs):
            writer.writerow([
                ep + 1,
                history["train_total"][ep],
                history["train_recon"][ep],
                history["train_kl"][ep],
                history["val_total"][ep],
                history["val_recon"][ep],
                history["val_kl"][ep],
            ])

    print(f"Saved loss history to: {csv_filename}")

    if return_history:
        return history


# ============================================================
# === CLI =====================================================
# ============================================================

def get_args():
    parser = argparse.ArgumentParser(description="VAE2D Training")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.0)

    return parser.parse_args()


# ============================================================
# === Main =====================================================
# ============================================================

if __name__ == "__main__":
    args = get_args()
    
    config = {
        "model": {
            "input_channels": 15,
            "hidden_channels": [32, 64, 128, 256, 512],
            "latent_dim": args.latent_dim,
            "input_size": 64,
            "kernel_size": 3,
            "activation": "relu",
            "batch_norm": True,
            "dropout_rate": 0.0,
            "weight_init": "kaiming_normal",
            "output_activation": "sigmoid",
            "use_reparameterization": True,
        }
    }
    train_vae2d(
        config,
        dataset_path="/pscratch/sd/n/ndwang/frequency_maps/frequency_maps_minmax.npy",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=5e-4,
        weight_decay=1e-4,
        max_steps=args.max_steps,
        B=args.beta
    )