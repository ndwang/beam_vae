import torch
from torch.utils.data import Dataset
import numpy as np

class FrequencyMapDataset(Dataset):
    def __init__(self, maps_path, scales_path, centroids_path=None, transform=None,
                 norm_stats=None):
        self.maps_path = maps_path
        self.scales_path = scales_path
        self.centroids_path = centroids_path
        self.transform = transform

        # Normalization stats: z-score per dimension for scales (in log-space) and centroids.
        # When provided, __getitem__ returns normalized values.
        self.norm_stats = norm_stats

        self._maps = None
        self._scales = None
        self._centroids = None

        # Validate shapes from a temporary mmap (cheap, no full load)
        maps_tmp = np.load(maps_path, mmap_mode='r')
        scales_tmp = np.load(scales_path, mmap_mode='r')
        self._len = maps_tmp.shape[0]
        if self._len != scales_tmp.shape[0]:
            raise ValueError(
                f"maps ({self._len}) and scales ({scales_tmp.shape[0]}) "
                "must have the same number of samples"
            )

    def _open_mmaps(self):
        if self._maps is None:
            self._maps = np.load(self.maps_path, mmap_mode='r')
            self._scales = np.load(self.scales_path, mmap_mode='r')
            if self.centroids_path is not None:
                self._centroids = np.load(self.centroids_path, mmap_mode='r')
            else:
                self._centroids = np.zeros((self._len, 6), dtype=np.float32)

    def __getstate__(self):
        """Strip mmap arrays before pickling so forkserver workers
        only receive lightweight file paths, not the full data."""
        state = self.__dict__.copy()
        state['_maps'] = None
        state['_scales'] = None
        state['_centroids'] = None
        return state

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        self._open_mmaps()
        maps = torch.from_numpy(self._maps[idx].copy()).float()
        scales = torch.from_numpy(self._scales[idx].copy()).float()
        centroids = torch.from_numpy(self._centroids[idx].copy()).float()

        if self.norm_stats is not None:
            scales = (torch.log(scales) - self.norm_stats['scale_mean']) / self.norm_stats['scale_std']
            centroids = (centroids - self.norm_stats['centroid_mean']) / self.norm_stats['centroid_std']

        if self.transform:
            maps = self.transform(maps)

        return maps, scales, centroids
