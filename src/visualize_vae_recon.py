import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from VAE2D import VAE2D
from ResidualVAE2D import ResidualVAE2D
from FrequencyMapDataset import FrequencyMapDataset
import joblib

min_max_scaler = joblib.load("/pscratch/sd/n/ndwang/frequency_maps/minmax_log_scaler.pkl")

def inverse_transform_images(data, mm_scaler):
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()

    data_flat = data.reshape(-1, 1)
    data_flat = mm_scaler.inverse_transform(data_flat)
    data_flat = np.exp(data_flat) - 1e-8

    return data_flat.reshape(data.shape)

def reconstruct(model, input_tensor, device, sample_idx=0):
    with torch.no_grad():
        inputs = input_tensor.unsqueeze(0).to(device, non_blocking=True)    
        recon, mu, logvar = model(inputs)
    return recon.squeeze(0).cpu().numpy()

def plot_vae_recon(target, recon, sample_idx=0, channel_indices=None, save_path="visualize_sample.png"):
    # Select channels to plot
    num_channels, height, width = target.shape
    if channel_indices == None:
        channel_indices = list(range(num_channels))
    num_plots = len(channel_indices)

    # Set up subplots and scaling
    fig_width = max(12, num_plots * 1.5)
    fig_height = 6
    figsize = (fig_width, fig_height)
    fig, axes = plt.subplots(3, num_plots, figsize=figsize)
    if num_plots == 1:
        axes = axes.reshape(-1, 1)

    # Compute abs error
    abs_error = [np.abs(target[channel] - recon[channel]) for channel in channel_indices]
    abs_error_max = np.max(abs_error)

    for col_idx, channel_idx in enumerate(channel_indices):
        target_channel = target[channel_idx]
        recon_channel = recon[channel_idx]
        abs_error_channel = abs_error[col_idx]

        
        vmin = target_channel.min()
        vmax = target_channel.max()
        # vmin = min(target_channel.min(), recon_channel.min())
        # vmax = max(target_channel.max(), recon_channel.max())

        # target
        im1 = axes[0, col_idx].imshow(target_channel, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, col_idx].set_title(f'Ch{channel_idx} target', fontsize=9)
        axes[0, col_idx].set_xticks([])
        axes[0, col_idx].set_yticks([])

        # recon
        im2 = axes[1, col_idx].imshow(recon_channel, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        axes[1, col_idx].set_title(f'Ch{channel_idx} recon', fontsize=9)
        axes[1, col_idx].set_xticks([])
        axes[1, col_idx].set_yticks([])
        
        # abs error
        mse_val = np.mean(abs_error_channel**2)
        im3 = axes[2, col_idx].imshow(abs_error_channel, cmap='viridis', vmin=0, vmax=abs_error_max, origin='lower')
        axes[2, col_idx].set_title(f'Ch{channel_idx} Abs Err\nMSE: {mse_val:.2e}', fontsize=9)
        axes[2, col_idx].set_xticks([])
        axes[2, col_idx].set_yticks([])
        
    # cbar1 = plt.colorbar(im1, ax=axes[0, :], location='left', shrink=0.6)
    # cbar1.ax.tick_params(labelsize=6)
    # cbar2 = plt.colorbar(im2, ax=axes[1, :], location='left', shrink=0.6)
    # cbar2.ax.tick_params(labelsize=6)
    # cbar3 = plt.colorbar(im3, ax=axes[2, :], location='left', shrink=0.6)
    # cbar3.ax.tick_params(labelsize=6)
    
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")
    
def main():
    parser = argparse.ArgumentParser(
        description='Visualize VAE reconstructions compared to original inputs'
    )    

    parser.add_argument('model', type=str, help='Path to the model .pth')
    parser.add_argument('dataset', type=str, help='Path to the dataset .npy')
    parser.add_argument('--sample-index', type=int, default=0, help='Index of the sample to visualize')
    parser.add_argument('--channels', type=int, nargs='+', default=None, help='Channels to plot (default all)')
    parser.add_argument('--log', action='store_true', help='Is the dataset log?') 

    args = parser.parse_args()
    sample_idx = args.sample_index
    channel_indices=args.channels
    
    config = {
            "model": {
                "input_channels": 15,
                "hidden_channels": [32, 64, 128, 256, 512],
                "latent_dim": 8,
                "input_size": 64,
                "kernel_size": 3,
                "activation": "relu",
                "batch_norm": True,
                "dropout_rate": 0.0,
                "weight_init": "kaiming_normal",
                "output_activation": "sigmoid",
                "use_reparameterization": False,
            }
        }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.model
    model = VAE2D(config)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    dataset = FrequencyMapDataset(args.dataset)
    target = dataset[sample_idx]

    print(f"Recosntructing sample{sample_idx:03d}")
    recon = reconstruct(model, target, device, sample_idx)

    target_orig = inverse_transform_images(target, min_max_scaler)
    recon_orig = inverse_transform_images(recon, min_max_scaler)

    if args.log:
        plot_vae_recon(target_orig, recon_orig, sample_idx, channel_indices, save_path=f"visualize_sample{sample_idx:03d}.png")
    else:
        plot_vae_recon(target.numpy(), recon, sample_idx, channel_indices, save_path=f"visualize_sample{sample_idx:03d}_latent{config['model']['latent_dim']}.png")
        
if __name__ == '__main__':
    main()