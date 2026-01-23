import time
from FrequencyMapDataset import FrequencyMapDataset
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from VAE2D import VAE2D
import torch
import torch.nn as nn
import torch.optim as optim

def benchmark_batch_size(num_workers=6):
    device = torch.device('cuda')

    data_dir = Path("/pscratch/sd/n/ndwang/frequency_maps")
    data_filename = "frequency_maps_minmax.npy"
    data_path = str(data_dir / data_filename)
    dataset = FrequencyMapDataset(data_path)

    config = {
        'model': {
            'input_channels': 15,
            'hidden_channels': [32, 64, 128, 256],
            'latent_dim': 64,
            'input_size': 64,
            'kernel_size': 3,
            'activation': 'relu',
            'batch_norm': True,
            'dropout_rate': 0.0,
            'weight_init': 'kaiming_normal',
            'use_reparameterization': True
        }
    }
    model = VAE2D(config)
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    results = []

    for bs in batch_sizes:
        print(f"\nTesting batch_size={bs}...")

        try:
            loader = DataLoader(
                dataset,
                batch_size=bs,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
            )

            total_samples = 0
            start_time = time.time()

            num_batches = 20

            for i, batch_data in enumerate(loader):
                if i >= num_batches:
                    break

                inputs = batch_data.to(device)
                targets = inputs

                optimizer.zero_grad()
                recon, mu, logvar = model(inputs)
                recon_loss = criterion(recon, targets)
                recon_loss.backward()
                optimizer.step()

                total_samples += bs

            end_time = time.time()
            duration = end_time - start_time
            throughput = total_samples / duration
            print(f"  Throughput: {throughput:.2f} samples/sec")

            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            mem_usage = f"{mem_alloc:.2f} GB (Alloc), {mem_reserved:.2f} GB (Res)"
            print(f"  Memory: {mem_usage}")
            results.append({
                'batch_size': bs,
                'throughput': throughput,
                'memory': mem_usage
            })

            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM: Out of Memory at batch_size={bs}")
                torch.cuda.empty_cache()
                break
            else:
                print(f"  Failed with error: {e}")
                break

        except Exception as e:
            print(f"  Failed with error: {e}")
            break

    if results:
        best_result = max(results, key=lambda x: x['throughput'])
        print("\n" + "="*40)
        print(f"Highest Throughput Batch Size: {best_result['batch_size']}")
        print(f"Max Throughput: {best_result['throughput']:.2f} samples/sec")
        print("="*40)
        
        print("\nFull Results:")
        print(f"{'Batch Size':<12} {'Throughput (samples/s)':<25} {'Memory':<20}")
        print("-" * 60)
        for r in results:
            print(f"{r['batch_size']:<12} {r['throughput']:<25.2f} {r['memory']:<20}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find optimal batch size for training')
    parser.add_argument('--num-workers', type=int, default=6, help='Number of workers to use for data loading during benchmark')
    
    args = parser.parse_args()
    
    benchmark_batch_size(args.num_workers)