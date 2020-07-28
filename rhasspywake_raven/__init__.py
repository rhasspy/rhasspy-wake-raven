"""Implementation of Snips Personal Wake Word Detector."""
import logging
import math
import time
import typing
from dataclasses import dataclass

import numpy as np
import python_speech_features
import scipy.io.wavfile
from rhasspysilence import WebRtcVadRecorder

from .dtw import DynamicTimeWarping

_LOGGER = logging.getLogger("rhasspy-wake-raven")

# -----------------------------------------------------------------------------


@dataclass
class Template:
    """Wakeword template."""

    duration_sec: float
    mfcc: np.ndarray
    name: str = ""

    @staticmethod
    def average_templates(
        templates: "typing.List[Template]", name: str = ""
    ) -> "Template":
        """Averages multiple templates piecewise into a single template.

        Credit to: https://github.com/mathquis/node-personal-wakeword
        """
        assert templates, "No templates"
        if len(templates) == 1:
            # Only one template
            return templates[0]

        # Use longest template as base
        templates = sorted(templates, key=lambda t: len(t.mfcc), reverse=True)
        base_template = templates[0]

        name = name or base_template.name
        base_mfcc: np.ndarray = base_template.mfcc
        rows, cols = base_mfcc.shape
        averages = [
            [[base_mfcc[row][col]] for col in range(cols)] for row in range(rows)
        ]

        avg_dtw = DynamicTimeWarping()

        # Collect features
        for template in templates[1:]:
            avg_dtw.compute_cost(template.mfcc, base_mfcc, keep_matrix=True)
            path = avg_dtw.compute_path()
            assert path is not None, "Failed to get DTW path"
            for row, col in path:
                for i, feature in enumerate(template.mfcc[row]):
                    averages[col][i].append(feature)

        # Average features
        avg_mfcc = np.array(
            [
                [np.mean(averages[row][col]) for col in range(cols)]
                for row in range(rows)
            ]
        )

        assert avg_mfcc.shape == base_mfcc.shape, "Wrong MFCC shape"

        return Template(
            duration_sec=base_template.duration_sec, mfcc=avg_mfcc, name=name
        )


# -----------------------------------------------------------------------------


class Raven:
    """
    Wakeword detector based on Snips Personal Wake Word Detector.
    https://medium.com/snips-ai/machine-learning-on-voice-a-gentle-introduction-with-snips-personal-wake-word-detector-133bd6fb568e

    Attributes
    ----------

    templates: List[Template]
        Wake word templates created from pre-trimmed WAV files

    probability_threshold: float = 0.5
        Probability above which which detection occurs

    minimum_matches: int = 0
        Minimum number of templates that must match for detection.
        Use 0 for all templates.

    distance_threshold: float = 0.22
        Cosine distance reference for probability calculation

    template_dtw: Optional[DynamicTimeWarping] = None
        DTW calculator (None for default)

    dtw_window_size: int = 5
        Size of Sakoe-Chiba window in DTW calculation

    dtw_step_pattern: float = 2
        Replacement cost multipler in DTW calculation

    shift_sec: float = DEFAULT_SHIFT_SECONDS
        Seconds to shift overlapping window by

    refractory_sec: float = 2
        Seconds after detection that new detection cannot occur

    skip_probability_threshold: float = 0.0
        Skip additional template calculations if probability is below this threshold

    failed_matches_to_refractory: Optional[int] = None
        Number of failed template matches before entering refractory period.
        Used to avoid high CPU usage and lag on low end systems.

    recorder: Optional[WebRtcVadRecorder] = None
        Silence detector (None for default settings).
        MFCC/DTW calculations are only done when a non-silent chunk of audio is
        detected. Calculations cease if at least N silence chunks are detected
        afterwards where N is half the number of chunks needed to span the
        average template duration. No calculations are done during refractory
        period.

    debug: bool = False
        If True, template probability calculations are logged
    """

    DEFAULT_SHIFT_SECONDS = 0.02

    def __init__(
        self,
        templates: typing.List[Template],
        keyword_name: str = "",
        probability_threshold: float = 0.5,
        minimum_matches: int = 0,
        distance_threshold: float = 0.22,
        template_dtw: typing.Optional[DynamicTimeWarping] = None,
        dtw_window_size: int = 5,
        dtw_step_pattern: float = 2,
        shift_sec: float = DEFAULT_SHIFT_SECONDS,
        refractory_sec: float = 2.0,
        skip_probability_threshold: float = 0.0,
        failed_matches_to_refractory: typing.Optional[int] = None,
        recorder: typing.Optional[WebRtcVadRecorder] = None,
        debug: bool = False,
    ):
        self.templates = templates
        assert self.templates, "No templates"

        self.keyword_name = keyword_name

        # Use or create silence detector
        self.recorder = recorder or WebRtcVadRecorder()
        self.vad_chunk_bytes = self.recorder.chunk_size
        self.sample_rate = self.recorder.sample_rate

        # Assume 16-bit samples
        self.sample_width = 2
        self.bytes_per_second = int(self.sample_rate * self.sample_width)

        # Match settings
        self.probability_threshold = probability_threshold
        self.minimum_matches = minimum_matches
        self.distance_threshold = distance_threshold
        self.skip_probability_threshold = skip_probability_threshold
        self.refractory_sec = refractory_sec
        self.failed_matches_to_refractory = failed_matches_to_refractory

        # Dynamic time warping calculation
        self.dtw = template_dtw or DynamicTimeWarping()
        self.dtw_window_size = dtw_window_size
        self.dtw_step_pattern = dtw_step_pattern

        # Average duration of templates
        template_duration_sec = sum([t.duration_sec for t in templates]) / len(
            templates
        )

        # Seconds to shift template window by during processing
        self.template_shift_sec = shift_sec
        self.shifts_per_template = (
            int(math.floor(template_duration_sec / shift_sec)) - 1
        )

        # Bytes needed for a template
        self.template_chunk_bytes = int(
            math.ceil(template_duration_sec * self.bytes_per_second)
        )

        # Ensure divisible by sample width
        while (self.template_chunk_bytes % self.sample_width) != 0:
            self.template_chunk_bytes += 1

        # Audio
        self.vad_audio_buffer = bytes()
        self.template_audio_buffer = bytes()
        self.example_audio_buffer = bytes()
        self.template_mfcc: typing.Optional[np.ndarray] = None
        self.template_chunks_left = 0
        self.num_template_chunks = int(
            math.ceil((self.template_chunk_bytes / self.vad_chunk_bytes) / 2)
        )

        # State machine
        self.num_refractory_chunks = int(
            math.ceil(
                self.sample_rate
                * self.sample_width
                * (refractory_sec / self.vad_chunk_bytes)
            )
        )
        self.refractory_chunks_left = 0
        self.failed_matches = 0
        self.match_seconds: typing.Optional[float] = None

        # If True, log DTW predictions
        self.debug = debug

        # Keep previously-computed distances and probabilities for debugging
        self.last_distances: typing.List[typing.Optional[float]] = [
            None for _ in self.templates
        ]

        self.last_probabilities: typing.List[typing.Optional[float]] = [
            None for _ in self.templates
        ]

    def process_chunk(self, chunk: bytes, keep_audio: bool = False) -> typing.List[int]:
        """Process a single chunk of raw audio data.

        Attributes
        ----------

        chunk: bytes
          Raw audio chunk

        Returns
        -------

        List of matching template indexes
        """
        self.vad_audio_buffer += chunk

        # Break audio into VAD-sized chunks (typically 30 ms)
        num_vad_chunks = int(
            math.floor(len(self.vad_audio_buffer) / self.vad_chunk_bytes)
        )
        if num_vad_chunks > 0:
            for i in range(num_vad_chunks):
                # Process single VAD-sized chunk
                matching_indexes = self._process_vad_chunk(i, keep_audio=keep_audio)
                if matching_indexes:
                    # Detection - reset and return immediately
                    self.vad_audio_buffer = bytes()
                    return matching_indexes

            # Remove processed audio
            self.vad_audio_buffer = self.vad_audio_buffer[
                (num_vad_chunks * self.vad_chunk_bytes) :
            ]

        # No detection
        return []

    def _process_vad_chunk(
        self, chunk_index: int, keep_audio: bool = False
    ) -> typing.List[int]:
        """Process the ith VAD-sized chunk of raw audio data from vad_audio_buffer.

        Attributes
        ----------

        chunk_index: int
            ith VAD-sized chunk in vad_audio_buffer

        Returns
        -------

        List of matching template indexes
        """
        matching_indexes: typing.List[int] = []

        if self.refractory_chunks_left > 0:
            self.refractory_chunks_left -= 1

            if self.refractory_chunks_left <= 0:
                _LOGGER.debug("Exiting refractory period")

            if keep_audio:
                self.example_audio_buffer = bytes()

            # In refractory period after wake word was detected.
            # Ignore any incoming audio.
            return matching_indexes

        # Test chunk for silence/speech
        chunk_start = chunk_index * self.vad_chunk_bytes
        chunk = self.vad_audio_buffer[chunk_start : chunk_start + self.vad_chunk_bytes]
        is_silence = self.recorder.is_silence(chunk)

        if is_silence:
            # Decrement audio chunks left to process before ignoring audio
            self.template_chunks_left = max(0, self.template_chunks_left - 1)
        else:
            # Reset count of audio chunks to process
            self.template_chunks_left = self.num_template_chunks

        if self.template_chunks_left <= 0:
            # No speech recently, so reset and ignore chunk.
            self._reset_state()

            if keep_audio:
                self.example_audio_buffer = bytes()

            return matching_indexes

        self.template_audio_buffer += chunk

        if keep_audio:
            self.example_audio_buffer += chunk

        # Process audio if there's enough for at least one template
        while len(self.template_audio_buffer) >= self.template_chunk_bytes:
            # Compute MFCC features for entire audio buffer (one or more templates)
            buffer_chunk = self.template_audio_buffer[: self.template_chunk_bytes]
            self.template_audio_buffer = self.template_audio_buffer[
                self.template_chunk_bytes :
            ]

            buffer_array = np.frombuffer(buffer_chunk, dtype=np.int16)

            mfcc_start_time = time.perf_counter()
            buffer_mfcc = python_speech_features.mfcc(
                buffer_array, winstep=self.template_shift_sec
            )
            if self.template_mfcc is None:
                # Brand new matrix
                self.template_mfcc = buffer_mfcc
            else:
                # Add to existing MFCC matrix
                self.template_mfcc = np.vstack((self.template_mfcc, buffer_mfcc))

            if self.debug:
                mfcc_end_time = time.perf_counter()
                _LOGGER.debug(
                    "MFCC for %s byte(s) in %s seconds",
                    len(buffer_chunk),
                    mfcc_end_time - mfcc_start_time,
                )

        last_row = (
            -1
            if (self.template_mfcc is None)
            else (len(self.template_mfcc) - self.shifts_per_template)
        )
        if last_row >= 0:
            assert self.template_mfcc is not None
            for row in range(last_row + 1):
                match_start_time = time.perf_counter()

                window_mfcc = self.template_mfcc[row : row + self.shifts_per_template]
                matching_indexes = self._process_window(window_mfcc)
                if matching_indexes:
                    # Clear buffers to avoid multiple detections and entire refractory period
                    self._reset_state()
                    self._begin_refractory()

                    # Record time for debugging
                    self.match_seconds = time.perf_counter() - match_start_time

                    return matching_indexes

                # Check for failure state
                self.failed_matches += 1
                if (self.failed_matches_to_refractory is not None) and (
                    self.failed_matches >= self.failed_matches_to_refractory
                ):
                    # Enter refractory period after too many failed template matches in a row
                    self._reset_state()
                    self._begin_refractory()
                    return matching_indexes

            self.template_mfcc = self.template_mfcc[last_row + 1 :]

        # No detections
        return matching_indexes

    def _process_window(self, window_mfcc: np.ndarray) -> typing.List[int]:
        """Process a single template-sized window of MFCC features.

        Returns
        -------

        List of matching template indexes
        """
        matching_indexes: typing.List[int] = []

        for i, template in enumerate(self.templates):
            # Compute optimal distance with a window
            dtw_start_time = time.perf_counter()

            distance = self.dtw.compute_cost(
                template.mfcc,
                window_mfcc,
                self.dtw_window_size,
                step_pattern=self.dtw_step_pattern,
            )

            # Normalize by sum of temporal dimensions
            normalized_distance = distance / (len(window_mfcc) + len(template.mfcc))

            # Compute detection probability
            probability = self.distance_to_probability(normalized_distance)

            if self.debug:
                dtw_end_time = time.perf_counter()
                _LOGGER.debug(
                    "%s %s: prob=%s, norm_dist=%s, dist=%s, dtw_time=%s, template_time=%s",
                    self.keyword_name,
                    i,
                    probability,
                    normalized_distance,
                    distance,
                    dtw_end_time - dtw_start_time,
                    template.duration_sec,
                )

            # Keep calculations results for debugging
            self.last_distances[i] = normalized_distance
            self.last_probabilities[i] = probability

            if probability >= self.probability_threshold:
                # Detection occured
                matching_indexes.append(i)

                if (self.minimum_matches > 0) and (
                    len(matching_indexes) >= self.minimum_matches
                ):
                    # Return immediately once minimum matches are satisfied
                    return matching_indexes
            elif probability < self.skip_probability_threshold:
                # Skip other templates if below threshold
                return matching_indexes

        return matching_indexes

    def _reset_state(self):
        """Reset VAD state machine."""
        self.template_audio_buffer = bytes()
        self.template_mfcc = None
        self.failed_matches = 0

    def _begin_refractory(self):
        """Enter refractory state where audio is ignored."""
        self.refractory_chunks_left = self.num_refractory_chunks
        _LOGGER.debug("Enter refractory for %s second(s)", self.refractory_sec)

    # -------------------------------------------------------------------------

    def distance_to_probability(self, normalized_distance: float) -> float:
        """Compute detection probability using distance and threshold."""
        return 1 / (
            1
            + math.exp(
                (normalized_distance - self.distance_threshold)
                / self.distance_threshold
            )
        )

    @staticmethod
    def wav_to_template(
        wav_file, name: str = "", shift_sec: float = DEFAULT_SHIFT_SECONDS
    ) -> Template:
        """Convert pre-trimmed WAV file to wakeword template."""
        sample_rate, wav_data = scipy.io.wavfile.read(wav_file)
        duration_sec = len(wav_data) / sample_rate
        wav_mfcc = python_speech_features.mfcc(wav_data, sample_rate, winstep=shift_sec)

        return Template(name=name, duration_sec=duration_sec, mfcc=wav_mfcc)
