import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
search_dir = "saved_models/beta_scan"
file_pattern = "loss_history_VAE2D_*.csv"
output_image = "KL_loss_curves_vs_beta.png"

# 1. Which parameter distinguishes the lines? (The one you are scanning)
# Options: 'latent', 'lr', 'B'
vary_param = 'B' 

# 2. Filter the other parameters to keep the comparison fair
# Set values to None to ignore, or specific numbers to filter.
filters = {
    # 'lr': 0.001,  # Example: Only look at runs with lr=0.001
    # 'B': 1e5,     # Example: Only look at runs with B=1e5
}

# 3. Which loss column to plot?
# Indices: 0:epoch, 1:tr_tot, 2:tr_rec, 3:tr_kl, 4:val_tot, 5:val_rec, 6:val_kl
col_idx = 6       # 4 = Validation Total Loss
y_label = "Validation KL Loss"

# --- Regex Setup ---
# Handles scientific notation for B and lr (e.g. 1e-05, 1.5e+02)
regex_pattern = re.compile(
    r"loss_history_VAE2D_e(?P<epochs>\d+)_B(?P<B>[\d\.e\+\-]+)_lr(?P<lr>[\d\.e\+\-]+)_latent(?P<latent>\d+)_MSE\.csv"
)

plot_data = []

print("Scanning files...")
files = glob.glob(os.path.join(search_dir, file_pattern))

for filepath in files:
    filename = os.path.basename(filepath)
    match = regex_pattern.search(filename)
    
    if match:
        params = match.groupdict()
        
        # Convert types
        current_params = {
            'epochs': int(params['epochs']),
            'B': float(params['B']),
            'latent': int(params['latent']),
            'lr': float(params['lr'])
        }

        # --- Filtering ---
        skip = False
        for key, value in filters.items():
            if value is not None:
                # Use isclose for float comparison to avoid precision errors
                if isinstance(value, float) or isinstance(current_params[key], float):
                    if not np.isclose(current_params[key], value):
                        skip = True; break
                elif current_params[key] != value:
                    skip = True; break
        if skip: continue

        # --- Reading Data ---
        try:
            # Read CSV, skip header
            data = np.loadtxt(filepath, delimiter=',', skiprows=1)
            
            # If file only has 1 epoch, data is 1D. We need 2D for consistent indexing.
            if data.ndim == 1:
                data = data.reshape(1, -1)
                
            # Extract Epochs (col 0) and Selected Loss (col_idx)
            epochs = data[:, 0]
            loss_values = data[:, col_idx]
            
            plot_data.append({
                'label_val': current_params[vary_param],
                'epochs': epochs,
                'loss': loss_values
            })
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# --- Plotting ---
if not plot_data:
    print("No matching data found to plot.")
else:
    # Sort data so the legend is in order (e.g. latent 32, 64, 128...)
    plot_data.sort(key=lambda x: x['label_val'])

    plt.figure(figsize=(10, 6))

    for item in plot_data:
        # Create a label for the legend (e.g., "latent=64")
        lbl = f"{vary_param}={item['label_val']}"
        
        plt.plot(item['epochs'], item['loss'], label=lbl, linewidth=2, alpha=0.8)

    plt.title(f'{y_label} vs Epochs\n(Varying {vary_param})')
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Optional: Log scale usually looks better for loss curves
    # plt.yscale('log') 

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Comparison plot saved to {output_image}")