"""Implementation of dynamic time warping.

Based on: https://github.com/mathquis/node-personal-wakeword
"""
import math
import typing

import numpy as np
import scipy.spatial.distance


class DynamicTimeWarping:
    """Computes DTW and holds results.

    Uses cosine distance.
    """

    def __init__(self,):
        self.cost_matrix: typing.Optional[np.ndarray] = None
        self.distance: typing.Optional[float] = None
        self.distance_func = scipy.spatial.distance.cosine

    def compute_cost(
        self, x: np.ndarray, y: np.ndarray, window: typing.Optional[int] = None
    ) -> np.ndarray:
        """Compute non-normalized distance between x and y with an optional window."""
        if window is None:
            return self._compute_optimal_path(x, y)

        return self._compute_optimal_path_with_window(x, y, window)

    def compute_path(self) -> typing.Optional[typing.List[typing.Tuple[int, int]]]:
        """Get actual path if cost matrix is available."""
        if self.cost_matrix is None:
            return None

        m, n = self.cost_matrix.shape
        row = m - 1
        col = n - 1
        path = [(row, col)]
        eps = 1e-14

        while (row > 0) or (col > 0):
            if (row > 0) and (col > 0):
                min_cost = min(
                    self.cost_matrix[row - 1][col],  # insertion
                    self.cost_matrix[row][col - 1],  # deletion
                    self.cost_matrix[row - 1][col - 1],  # match
                )

                if math.isclose(
                    min_cost, self.cost_matrix[row - 1][col - 1], rel_tol=eps
                ):
                    row = row - 1
                    col = col - 1
                elif math.isclose(
                    min_cost, self.cost_matrix[row - 1][col], rel_tol=eps
                ):
                    row = row - 1
                elif math.isclose(
                    min_cost, self.cost_matrix[row][col - 1], rel_tol=eps
                ):
                    col = col - 1
            elif (row > 0) and (col == 0):
                row = row - 1
            elif (row == 0) and (col > 0):
                col = col - 1

            path.append((row, col))

        return list(reversed(path))

    # -------------------------------------------------------------------------

    def _compute_optimal_path(self, x: np.ndarray, y: np.ndarray) -> float:
        """Computes optimal path between x and y."""
        m = len(x)
        n = len(y)

        cost_matrix = np.full(shape=(m, n), fill_value=math.inf, dtype=float)
        cost_matrix[0][0] = self.distance_func(x[0], y[0])

        for row in range(1, m):
            cost = self.distance_func(x[row], y[0])
            cost_matrix[row][0] = cost + cost_matrix[row - 1][0]

        for col in range(1, n):
            cost = self.distance_func(x[0], y[col])
            cost_matrix[0][col] = cost + cost_matrix[0][col - 1]

        for row in range(1, m):
            for col in range(1, n):
                cost = self.distance_func(x[row], y[col])
                cost_matrix[row][col] = cost + min(
                    cost_matrix[row - 1][col],  # insertion
                    cost_matrix[row][col - 1],  # deletion
                    cost_matrix[row - 1][col - 1],  # match
                )

        self.cost_matrix = cost_matrix
        distance = self.cost_matrix[m - 1][n - 1]
        self.distance = distance

        return distance

    def _compute_optimal_path_with_window(
        self, x: np.ndarray, y: np.ndarray, window: int
    ) -> float:
        """Computes optimal path between x and y using a window."""
        m = len(x)
        n = len(y)

        window = max(window, abs(m - n))
        cost_matrix = np.full(shape=(m + 1, n + 1), fill_value=math.inf, dtype=float)
        cost_matrix[0][0] = 0

        for row in range(1, m + 1):
            col_start = max(1, row - window)
            col_end = min(n, row + window)

            for col in range(col_start, col_end + 1):
                cost = self.distance_func(x[row - 1], y[col - 1])
                cost_matrix[row][col] = cost + min(
                    cost_matrix[row - 1][col],  # insertion
                    cost_matrix[row][col - 1],  # deletion
                    cost_matrix[row - 1][col - 1],  # match
                )

        self.cost_matrix = cost_matrix[1:, 1:]
        distance = self.cost_matrix[m - 1][n - 1]
        self.distance = distance

        return distance
