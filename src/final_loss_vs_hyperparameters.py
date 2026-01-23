import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
search_dir = "."  # Directory containing the csv files
file_pattern = "loss_history_VAE2D_*.csv"
output_image = "final_loss_vs_latent.png"

# 1. Choose the hyperparameter for the X-axis
# Options based on your filename: 'latent', 'lr', 'B', 'epochs'
target_param = 'latent' 

# 2. Filter other parameters to ensure a clean curve
# (e.g., if plotting against 'latent', you usually want 'lr' and 'B' to be constant)
# Set value to None to include all.
filters = {
    'B': 0.0,
    # 'lr': 0.001,  # Example: Only keep runs where lr == 0.001
    # 'B': 64,      # Example: Only keep runs where B == 64
}

# Regex to parse the filename: 
# loss_history_VAE2D_e{epochs}_B{B}_lr{lr}_latent{latent}_MSE.csv
regex_pattern = re.compile(
    r"loss_history_VAE2D_e(?P<epochs>\d+)_B(?P<B>[\d\.e\+\-]+)_lr(?P<lr>[\d\.e\+\-]+)_latent(?P<latent>\d+)_MSE\.csv"
)

results = []

# --- Processing Files ---
files = glob.glob(os.path.join(search_dir, file_pattern))
print(f"Found {len(files)} files.")

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

        # Apply Filters
        skip = False
        for key, value in filters.items():
            if value is not None and current_params[key] != value:
                skip = True
                break
        if skip: continue

        try:
            # Read data using np.loadtxt
            # skiprows=1 to ignore the header row
            data = np.loadtxt(filepath, delimiter=',', skiprows=1)
            
            # Handle case where file might only have 1 row (1D array) vs multiple (2D array)
            if data.ndim == 1:
                last_row = data
            else:
                last_row = data[-1]

            # Columns based on your writerow order:
            # 0:epoch, 1:tr_tot, 2:tr_rec, 3:tr_kl, 4:val_tot, 5:val_rec, 6:val_kl
            
            entry = {
                'x': current_params[target_param],
                'val_total': last_row[4],
                'val_recon': last_row[5],
                'val_kl': last_row[6]
            }
            results.append(entry)
            
        except Exception as e:
            print(f"Skipping {filename}: {e}")

# --- Plotting ---
if not results:
    print("No matching data found.")
else:
    # Sort by x-axis value so the line connects correctly
    results.sort(key=lambda k: k['x'])

    x_vals = [r['x'] for r in results]
    y_total = [r['val_total'] for r in results]
    y_recon = [r['val_recon'] for r in results]
    
    plt.figure(figsize=(10, 6))
    
    # Plot Total Loss
    plt.plot(x_vals, y_total, marker='o', label='Final Val Total Loss', linewidth=2)
    
    # Plot Recon Loss (often similar to total, but good to see)
    plt.plot(x_vals, y_recon, marker='s', linestyle='--', label='Final Val Recon Loss', alpha=0.6)

    plt.title(f'Final Validation Loss vs {target_param}')
    plt.xlabel(target_param)
    plt.ylabel('Loss')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # Log scale for X axis if scanning Learning Rate
    if target_param == 'lr':
        plt.xscale('log')

    plt.tight_layout()
    
    # Save instead of show
    plt.savefig(output_image, dpi=300)
    print(f"Plot saved to {output_image}")