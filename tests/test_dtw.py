"""Test cases for dynamic time warping."""
import math
import unittest
from pathlib import Path

import scipy.io.wavfile
import numpy as np
import scipy.spatial.distance
import python_speech_features
from rhasspywake_raven.dtw import DynamicTimeWarping

_DIR = Path(__file__).parent

# -----------------------------------------------------------------------------


class DTWTests(unittest.TestCase):
    """Test cases for dynamic time warping."""

    def setUp(self):
        self.dtw = DynamicTimeWarping(distance_func="euclidean")

    def test_nowindow(self):
        """Test cost calculation without a window."""
        x = np.array([1, 2, 3])
        y = np.array([2, 2, 2, 3, 4])

        distance = self.dtw.compute_cost(x, y, keep_matrix=True)
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
        x = np.array([1, 2, 3, 3, 5])
        y = np.array([1, 2, 2, 2, 2, 2, 2, 4])

        distance = self.dtw.compute_cost(x, y, window=3, keep_matrix=True)
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

    def test_mfcc(self):
        """Test cost calculation with actual WAV files."""

        # Use cosine distance
        self.dtw = DynamicTimeWarping()

        rate_1, wav_1 = scipy.io.wavfile.read("etc/okay-rhasspy/okay-rhasspy-00.wav")
        rate_2, wav_2 = scipy.io.wavfile.read("etc/okay-rhasspy/okay-rhasspy-01.wav")
        rate_3, wav_3 = scipy.io.wavfile.read("etc/hey-mycroft/hey-mycroft-00.wav")

        mfcc_1 = python_speech_features.mfcc(wav_1, rate_1)
        mfcc_2 = python_speech_features.mfcc(wav_2, rate_2)
        mfcc_3 = python_speech_features.mfcc(wav_3, rate_3)

        # Verify both "okay rhasspy" templates match
        distance = self.dtw.compute_cost(mfcc_1, mfcc_2, window=5, step_pattern=2)
        normalized_distance = distance / (len(mfcc_1) + len(mfcc_2))

        # Compute detection probability
        probability = self._distance_to_probability(normalized_distance)

        # p >= 0.5
        self.assertGreaterEqual(probability, 0.5)

        # Verify "okay rhasspy" and "hey mycroft" templates don't match
        distance = self.dtw.compute_cost(mfcc_1, mfcc_3, window=5, step_pattern=2)
        normalized_distance = distance / (len(mfcc_1) + len(mfcc_3))

        # Compute detection probability
        probability = self._distance_to_probability(normalized_distance)

        # p < 0.5
        self.assertLess(probability, 0.5)

    def _distance_to_probability(
        self, normalized_distance: float, distance_threshold: float = 0.22
    ) -> float:
        """Compute detection probability using distance and threshold."""
        return 1 / (
            1
            + math.exp((normalized_distance - distance_threshold) / distance_threshold)
        )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
