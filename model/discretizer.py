import numpy as np

class MarkDiscretizer:
    def __init__(self, M, int_res, clip=True):
        """
        Args:
        - M:        [n_marks, 2] array of per-dim bounds [[low, high], ...]
        - int_res:  int, number of bins per dimension
        - clip:     bool, clip inputs to bounds before binning
        """
        self.M = np.asarray(M, dtype=float)          # [n_marks, 2]
        assert self.M.ndim == 2 and self.M.shape[1] == 2
        self.n_marks = self.M.shape[0]
        self.int_res = int_res
        self.clip = clip
        # Precompute scales to turn values into bin indices quickly
        lows  = self.M[:, 0]
        highs = self.M[:, 1]
        assert np.all(highs > lows), "Each dim must have high > low."
        self._lows  = lows
        self._highs = highs
        self._width = highs - lows
        # bin index ∈ {0,...,int_res-1}; scale = int_res / width
        self._scale = (int_res / self._width)
        # shape for flattening/unflattening
        self._shape = (int_res,) * self.n_marks

    def _to_bins(self, marks):
        """Return per-dim integer bin indices [batch_size, n_marks]."""
        x = np.asarray(marks, dtype=float)
        assert x.ndim == 2 and x.shape[1] == self.n_marks, f"marks must be [batch_size, n_marks], found {x.shape}"
        if self.clip:
            x = np.minimum(np.maximum(x, self._lows), self._highs)
        # map to [0, int_res); careful with the right edge: put highs → int_res-1
        # compute float bin positions
        pos = (x - self._lows) * self._scale
        bins = np.floor(pos).astype(np.int64)
        # handle exact-right-edge and any numeric drift
        bins = np.clip(bins, 0, self.int_res - 1)
        return bins

    def fit_transform(self, marks):
        """
        Args:
        - marks:   [ batch_size, n_marks ] np.ndarray (continuous)
        Outs:
        - indices: [ batch_size ] np.ndarray (flattened grid cell ids)
        """
        bins = self._to_bins(marks)                            # [ B, n_marks ]
        indices = np.ravel_multi_index(bins.T, self._shape)    # [ B ]
        return indices

    def transform(self, marks):
        """Use stored bounds to discretize new data."""
        return self.fit_transform(marks)    # [ batch_size ]

    def inverse_transform_to_bin_centers(self, indices):
        """
        Map flat indices back to bin-center coordinates in continuous space.
        Args:
        - indices: [batch_size] integer flat ids
        Outs:
        - centers: [batch_size, n_marks] float bin centers
        """
        indices = np.asarray(indices, dtype=np.int64)
        multi = np.column_stack(np.unravel_index(indices, self._shape))  # [B, n_marks]
        # bin centers: low + (k + 0.5) * width/int_res
        centers = self._lows + (multi + 0.5) * (self._width / self.int_res)
        return centers
    
if __name__ == '__main__':
    # Example: 3D marks each bounded in [0,1], 16 bins per dimension
    M = np.array([[0.,1.],
                [0.,1.],
                [0.,1.]])
    md = MarkDiscretizer(M, int_res=5, clip=True)

    # [ batch_size, n_marks ]
    marks = np.array([[0.1, 0.5, 0.9],
                    [0.0, 1.0, 1.0],
                    [0.9999, 0.0001, 0.5]])

    idx = md.fit_transform(marks)                       # [batch_size]
    centers = md.inverse_transform_to_bin_centers(idx)  # [batch_size, n_marks]