import time
from FrequencyMapDataset import FrequencyMapDataset
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

def benchmark_workers(batch_size=64, num_batches=100, max_workers=32):
    data_dir = Path("/pscratch/sd/n/ndwang/frequency_maps")
    data_filename = "frequency_maps_minmax.npy"
    data_path = str(data_dir / data_filename)
    dataset = FrequencyMapDataset(data_path)
        
    worker_counts = [0]
    worker_counts.extend([i for i in range(2, max_workers+1, 2)])
    
    print(f"Benchmarking workers for dataset: {data_path}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches to measure: {num_batches}")
    print(f"Worker counts to test: {worker_counts}")
    
    results = []
    
    for num_workers in worker_counts:
        print(f"\nTesting num_workers={num_workers}...")
    
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
    
        start_time = time.time()
        count = 0
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            count += 1
    
        end_time = time.time()
        duration = end_time - start_time
        throughput = (count * batch_size) / duration
    
        print(f"  Time: {duration:.4f}s")
        print(f"  Throughput: {throughput:.2f} samples/sec")
    
        results.append({
            'num_workers': num_workers,
            'duration': duration,
            'throughput': throughput
        })
    
    best_result = max(results, key=lambda x: x['throughput'])
    
    print("\n" + "="*40)
    print(f"Optimal num_workers: {best_result['num_workers']}")
    print(f"Max Throughput: {best_result['throughput']:.2f} samples/sec")
    print("="*40)
    
    print("\nFull Results:")
    print(f"{'Workers':<10} {'Time (s)':<10} {'Throughput (samples/s)':<25}")
    print("-" * 45)
    for r in results:
        print(f"{r['num_workers']:<10} {r['duration']:<10.4f} {r['throughput']:<25.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find optimal number of workers for DataLoader')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size to use (default: 64)')
    parser.add_argument('--num-batches', type=int, default=100, help='Number of batches to measure (default: 100)')
    parser.add_argument('--max-workers', type=int, default=32, help='Maximum number of workers to test (default: 32)')
    
    args = parser.parse_args()
    
    benchmark_workers(args.batch_size, args.num_batches, args.max_workers)
