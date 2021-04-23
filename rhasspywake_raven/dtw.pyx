"""Implementation of dynamic time warping.

Based on: https://github.com/mathquis/node-personal-wakeword
"""
import math
import typing

cimport numpy as cnp
cimport cython
import numpy as np
import scipy.spatial.distance

class DynamicTimeWarping:
    """Computes DTW and holds results.

    Uses cosine distance and sakoe-chiba constraint by default.
    """

    def __init__(self, distance_func: str = "cosine"):
        self.cost_matrix: typing.Optional[np.ndarray] = None
        self.distance: typing.Optional[float] = None
        self.distance_func = distance_func or "cosine"

    def compute_cost(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: typing.Optional[int] = None,
        **cost_args
    ) -> np.ndarray:
        """Compute non-normalized distance between x and y with an optional window."""
        if window is None:
            return self._compute_optimal_path(x, y, **cost_args)

        return self._compute_optimal_path_with_window(x, y, window, **cost_args)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_path(self) -> typing.Optional[typing.List[typing.Tuple[int, int]]]:
        """Get actual path if cost matrix is available."""
        cdef:
            cnp.double_t insertion, deletion, match
            Py_ssize_t m, n, row, col
            cnp.ndarray[cnp.double_t, ndim=2] cost_matrix

        if self.cost_matrix is None:
            return None

        cost_matrix = self.cost_matrix
        m = cost_matrix.shape[0]
        n = cost_matrix.shape[1]
        row = m - 1
        col = n - 1
        path = [(row, col)]

        while row or col:
            if row and col:
                insertion = cost_matrix[row - 1, col]
                deletion = cost_matrix[row, col - 1]
                match = cost_matrix[row - 1, col - 1]
 
                if match <= insertion and match <= deletion:
                    row -= 1
                    col -= 1
                elif insertion <= deletion:
                    row -= 1
                else:
                    col -= 1
            elif row:
                row -= 1
            elif col:
                col -= 1

            path.append((row, col))

        return list(reversed(path))

    # -------------------------------------------------------------------------

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _compute_optimal_path(
        self, x: np.ndarray, y: np.ndarray, keep_matrix=False
    ) -> float:
        """Computes optimal path between x and y."""
        cdef:
            Py_ssize_t row, col
            Py_ssize_t m, n
            cnp.double_t distance
            cnp.ndarray[cnp.double_t, ndim=2] cost_matrix

        m = len(x)
        n = len(y)

        # Need 2-D arrays for distance calculation
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        cost_matrix = scipy.spatial.distance.cdist(x, y, metric=self.distance_func)

        for row in range(1, m):
            cost_matrix[row, 0] += cost_matrix[row - 1, 0]

        for col in range(1, n):
            cost_matrix[0, col] += cost_matrix[0, col - 1]

        for row in range(1, m):
            for col in range(1, n):
                cost_matrix[row, col] += min(
                    cost_matrix[row - 1, col],  # insertion
                    cost_matrix[row, col - 1],  # deletion
                    cost_matrix[row - 1, col - 1],  # match
                )

        if keep_matrix:
            self.cost_matrix = cost_matrix

        distance = cost_matrix[m - 1, n - 1]
        self.distance = distance

        return distance

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _compute_optimal_path_with_window(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: int,
        step_pattern: float = 1,
        keep_matrix=False,
    ) -> float:
        """Computes optimal path between x and y using a window."""
        cdef:
            Py_ssize_t n, m, row, col, col_start, col_end, cwindow
            cnp.double_t cost, distance
            cnp.ndarray[cnp.double_t, ndim=2] distance_matrix
            cnp.ndarray[cnp.double_t, ndim=2] cost_matrix

        n = len(x)
        m = len(y)

        # Avoid case where endpoint lies outside band
        len_diff = m - n if m > n else n - m
        cwindow = max(<Py_ssize_t>window, len_diff)

        # Need 2-D arrays for distance calculation
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # Pre-compute distance between all pairs
        distance_matrix = scipy.spatial.distance.cdist(x, y, metric=self.distance_func)

        cost_matrix = np.full(shape=(n + 1, m + 1), fill_value=math.inf, dtype=float)

        cost_matrix[0, 0] = 0
        for row in range(1, n + 1):
            col_start = max(1, row - window)
            col_end = min(m, row + window)

            for col in range(col_start, col_end + 1):
                cost = distance_matrix[row - 1, col - 1]

                # symmetric step pattern
                cost_matrix[row, col] = min(
                    (step_pattern * cost) + cost_matrix[row - 1, col - 1],
                    cost + cost_matrix[row - 1, col],
                    cost + cost_matrix[row, col - 1],
                )

        if keep_matrix:
            self.cost_matrix = cost_matrix[1:, 1:]

        distance = cost_matrix[n, m]
        self.distance = distance

        return distance
