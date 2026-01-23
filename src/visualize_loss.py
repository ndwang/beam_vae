import csv
import matplotlib.pyplot as plt
import argparse
import os

def read_csv_data(filename):
    """
    Reads the CSV using the built-in csv module and returns a dictionary of lists.
    """
    data = {
        "epoch": [],
        "train_total": [], "train_recon": [], "train_kl": [],
        "val_total": [], "val_recon": [], "val_kl": []
    }

    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data["epoch"].append(int(row["epoch"]))
                data["train_total"].append(float(row["train_total"]))
                data["train_recon"].append(float(row["train_recon"]))
                data["train_kl"].append(float(row["train_kl"]))
                data["val_total"].append(float(row["val_total"]))
                data["val_recon"].append(float(row["val_recon"]))
                data["val_kl"].append(float(row["val_kl"]))
        return data
        
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except ValueError as e:
        print(f"Error parsing numbers in CSV: {e}")
        return None

def plot_data(data, filename, save_plot=False):
    """
    Plots the data dictionary on a Logarithmic Y-Scale.
    """
    if not data:
        return

    epochs = data['epoch']
    
    # Create 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Training History (Log Scale): {os.path.basename(filename)}', fontsize=16)

    # --- Plot 1: Total Loss ---
    axes[0].plot(epochs, data['train_total'], label='Train', color='blue')
    axes[0].plot(epochs, data['val_total'], label='Val', color='orange', linestyle='--')
    axes[0].set_title('Total Loss')

    # --- Plot 2: Reconstruction Loss ---
    axes[1].plot(epochs, data['train_recon'], label='Train', color='green')
    axes[1].plot(epochs, data['val_recon'], label='Val', color='red', linestyle='--')
    axes[1].set_title('Reconstruction Loss')

    # --- Plot 3: KL Divergence ---
    axes[2].plot(epochs, data['train_kl'], label='Train', color='purple')
    axes[2].plot(epochs, data['val_kl'], label='Val', color='brown', linestyle='--')
    axes[2].set_title('KL Divergence')

    # Common styling for all axes
    for ax in axes:
        ax.set_yscale('log')  # <--- SETS LOG SCALE
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (Log Scale)')
        ax.grid(True, which="both", ls="-", alpha=0.3) # Grid for log scale needs 'both'
        ax.legend()

    plt.tight_layout()

    if save_plot:
        out_filename = os.path.splitext(filename)[0] + ".png"
        plt.savefig(out_filename)
        print(f"Plot saved to: {out_filename}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot training losses from CSV (Log Scale).")
    parser.add_argument("filename", type=str, help="Path to the CSV file.")
    parser.add_argument("--save", action="store_true", help="Save plot as image instead of displaying.")
    
    args = parser.parse_args()

    data = read_csv_data(args.filename)
    if data:
        plot_data(data, args.filename, args.save)

if __name__ == "__main__":
    main()