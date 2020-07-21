"""Test cases for dynamic time warping."""
import math
import unittest

import numpy as np
import scipy.spatial.distance
from rhasspywake_raven.dtw import DynamicTimeWarping


class DTWTests(unittest.TestCase):
    """Test cases for dynamic time warping."""

    def setUp(self):
        self.dtw = DynamicTimeWarping(distance_func=scipy.spatial.distance.euclidean)

    def test_nowindow(self):
        """Test cost calculation without a window."""
        x = [1, 2, 3]
        y = [2, 2, 2, 3, 4]

        distance = self.dtw.compute_cost(x, y)
        self.assertEqual(distance, 2.0)
        self.assertTrue(
            np.array_equal(
                self.dtw.cost_matrix,
                np.array(
                    [[1, 2, 3, 5, 8], [1, 1, 1, 2, 4], [2, 2, 2, 1, 2]], dtype=float
                ),
            ),
            self.dtw.cost_matrix,
        )

    def test_window(self):
        """Test cost calculation with a window."""
        x = [1, 2, 3, 3, 5]
        y = [1, 2, 2, 2, 2, 2, 2, 4]

        distance = self.dtw.compute_cost(x, y, window=3)
        self.assertEqual(distance, 3.0)
        self.assertTrue(
            np.array_equal(
                self.dtw.cost_matrix,
                np.array(
                    [
                        [0, 1, 2, 3, math.inf, math.inf, math.inf, math.inf],
                        [1, 0, 0, 0, 0, math.inf, math.inf, math.inf],
                        [3, 1, 1, 1, 1, 1, math.inf, math.inf],
                        [5, 2, 2, 2, 2, 2, 2, math.inf],
                        [math.inf, 5, 5, 5, 5, 5, 5, 3],
                    ],
                    dtype=float,
                ),
            ),
            self.dtw.cost_matrix,
        )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
