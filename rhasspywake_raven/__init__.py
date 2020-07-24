"""Implementation of Snips Personal Wake Word Detector."""
import logging
import math
import time
import typing
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np
import python_speech_features
import scipy.io.wavfile
from rhasspysilence import WebRtcVadRecorder

from .dtw import DynamicTimeWarping

_LOGGER = logging.getLogger("rhasspy-wake-raven")


class RavenState(int, Enum):
    """State of Raven detector."""

    IN_SILENCE = 0
    IN_SPEECH = 1
    IN_REFRACTORY = 2


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

    frame_dtw: Optional[DynamicTimeWarping] = None
        DTW calculator (None for default)

    dtw_window_size: int = 5
        Size of Sakoe-Chiba window in DTW calculation

    dtw_step_pattern: float = 2
        Replacement cost multipler in DTW calculation

    sample_rate: int = 16000
        Sample rate of audio chunks in Hertz.

    chunk_size: int = 960
        Size of audio chunks in bytes.
        Must be 10, 20, or 30 ms for VAD calculation.
        A sample width of 2 bytes (16 bits) is assumed.

    shift_sec: float = 0.01
        Seconds to shift overlapping window by

    before_chunks: int = 0
        Chunks of audio before speech to keep in window

    refractory_sec: float = 2
        Skip additional template calculations if probability is below this threshold

    refractory_sec: float = 2
        Seconds after detection that new detection cannot occur

    recorder: Optional[WebRtcVadRecorder] = None
        Silence detector (None for default settings).
        MFCC/DTW calculations are only done when a non-silent chunk of audio is
        detected. Calculations cease if at least N silence chunks are detected
        afterwards where N is the number of chunks needed to span the average
        template duration. No calculations are done during refractory period.

    debug: bool = False
        If True, template probability calculations are logged
    """

    def __init__(
        self,
        templates: typing.List[Template],
        probability_threshold: float = 0.5,
        minimum_matches: int = 0,
        distance_threshold: float = 0.22,
        frame_dtw: typing.Optional[DynamicTimeWarping] = None,
        dtw_window_size: int = 5,
        dtw_step_pattern: float = 2,
        sample_rate: int = 16000,
        chunk_size: int = 960,
        shift_sec: float = 0.01,
        before_chunks: int = 0,
        refractory_sec: float = 2.0,
        skip_probability_threshold: float = 0.0,
        recorder: typing.Optional[WebRtcVadRecorder] = None,
        debug: bool = False,
    ):
        self.templates = templates
        assert self.templates, "No templates"

        self.probability_threshold = probability_threshold
        self.minimum_matches = minimum_matches
        self.distance_threshold = distance_threshold
        self.skip_probability_threshold = skip_probability_threshold

        self.chunk_size = chunk_size
        self.shift_sec = shift_sec
        self.sample_rate = sample_rate

        self.before_buffer: typing.Optional[typing.Deque[bytes]] = None
        if before_chunks > 0:
            self.before_buffer = deque(maxlen=before_chunks)

        # Assume 16-bit samples
        self.sample_width = 2
        self.chunk_seconds = (self.chunk_size / self.sample_width) / self.sample_rate

        # Use or create silence detector
        self.recorder = recorder or WebRtcVadRecorder()

        # Dynamic time warping calculation
        self.dtw = frame_dtw or DynamicTimeWarping()
        self.dtw_window_size = dtw_window_size
        self.dtw_step_pattern = dtw_step_pattern

        # Keep previously-computed distances and probabilities for debugging
        self.last_distances: typing.List[typing.Optional[float]] = [
            None for _ in self.templates
        ]

        self.last_probabilities: typing.List[typing.Optional[float]] = [
            None for _ in self.templates
        ]

        # Average duration of templates
        self.frame_duration_sec = sum([t.duration_sec for t in templates]) / len(
            templates
        )

        # Size in bytes of a frame
        self.window_chunk_size = (
            self.seconds_to_chunks(self.frame_duration_sec)
            * self.chunk_size
            * self.sample_width
        )

        # Ensure divisible by sample width
        while (self.window_chunk_size % self.sample_width) != 0:
            self.window_chunk_size += 1

        # Size in bytes to shift each frame.
        # Should be less than the size of a frame to ensure overlap.
        self.shift_size = (
            self.seconds_to_chunks(self.shift_sec) * self.chunk_size * self.sample_width
        )

        # State machine
        self.audio_buffer = bytes()
        self.state = RavenState.IN_SILENCE
        self.num_silence_chunks = self.seconds_to_chunks(self.frame_duration_sec)
        self.silence_chunks_left = 0
        self.num_refractory_chunks = self.seconds_to_chunks(refractory_sec)
        self.refractory_chunks_left = 0
        self.match_seconds: typing.Optional[float] = None

        self.debug = debug

    def process_chunk(self, chunk: bytes) -> typing.List[int]:
        """Process a single chunk of raw audio data.

        Must be the same length as self.chunk_size.

        Attributes
        ----------

        chunk: bytes
          Raw audio chunk with one or more windows

        Returns
        -------

        List of matching template indexes
        """
        assert (
            len(chunk) == self.chunk_size
        ), f"Expected chunk length to be {self.chunk_size}, got {len(chunk)}"

        if self.state == RavenState.IN_REFRACTORY:
            self.refractory_chunks_left -= 1
            if self.refractory_chunks_left > 0:
                # In refractory period
                return []

            # Done with refractory period
            self.state = RavenState.IN_SILENCE
            self.audio_buffer = bytes()

            if self.before_buffer is not None:
                self.before_buffer.clear()

        # Keep chunks before detection
        if self.before_buffer is not None:
            self.before_buffer.append(chunk)

        # Test chunk for silence/speech
        vad_start_time = time.perf_counter()
        is_silence = self.recorder.is_silence(chunk)

        if self.debug:
            vad_end_time = time.perf_counter()
            _LOGGER.debug("VAD on chunk in %s second(s)", vad_end_time - vad_start_time)

        if self.state == RavenState.IN_SILENCE:
            # Only silence so far
            if not is_silence:
                # Start recording and checking audio
                self.audio_buffer = bytes()

                # Include some chunks before speech
                if self.before_buffer is not None:
                    for before_chunk in self.before_buffer:
                        self.audio_buffer += before_chunk

                    self.before_buffer.clear()

                self.state = RavenState.IN_SPEECH
                self.silence_chunks_left = self.num_silence_chunks
        else:
            # Speech recently
            if is_silence:
                # Decrement chunk count and go back to sleep if there's too many
                # silent chunks in a row.
                self.silence_chunks_left = self.silence_chunks_left - 1
                if self.silence_chunks_left <= 0:
                    # Back to sleep
                    self.state = RavenState.IN_SILENCE
            else:
                # Reset silence chunks
                self.silence_chunks_left = self.num_silence_chunks

        if self.state == RavenState.IN_SPEECH:
            match_start_time = time.perf_counter()

            # Include audio for processing
            self.audio_buffer += chunk

            # Process all frames in buffer
            while len(self.audio_buffer) >= self.window_chunk_size:
                frame_chunk = self.audio_buffer[: self.window_chunk_size]

                # Shift buffer into overlapping frame
                self.audio_buffer = self.audio_buffer[self.shift_size :]

                # Process single frame
                frame = np.frombuffer(frame_chunk, dtype=np.int16)

                matching_indexes = self._process_frame(frame)
                if matching_indexes:
                    # Reset state to avoid multiple detections.
                    self.state = RavenState.IN_REFRACTORY
                    self.refractory_chunks_left = self.num_refractory_chunks
                    self.match_seconds = time.perf_counter() - match_start_time

                    return matching_indexes

        # No detections
        return []

    def _process_frame(self, frame: np.ndarray) -> typing.List[int]:
        """Process a single overlapping frame of audio data.

        Returns
        -------

        List of matching template indexes
        """
        matching_indexes: typing.List[int] = []

        mfcc_start_time = time.perf_counter()
        frame_mfcc = python_speech_features.mfcc(frame, self.sample_rate)
        if self.debug:
            mfcc_end_time = time.perf_counter()
            _LOGGER.debug(
                "MFCC on frame in %s second(s)", mfcc_end_time - mfcc_start_time
            )

        for i, template in enumerate(self.templates):
            # Compute optimal distance with a window
            dtw_start_time = time.perf_counter()
            distance = self.dtw.compute_cost(
                template.mfcc,
                frame_mfcc,
                self.dtw_window_size,
                step_pattern=self.dtw_step_pattern,
            )

            # Normalize by sum of temporal dimensions
            normalized_distance = distance / (len(frame_mfcc) + len(template.mfcc))

            # Compute detection probability
            probability = self.distance_to_probability(normalized_distance)

            if self.debug:
                dtw_end_time = time.perf_counter()
                _LOGGER.debug(
                    "Template %s: prob=%s, norm_dist=%s, dist=%s, dtw_time=%s, template_time=%s",
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

    def seconds_to_chunks(self, seconds: float) -> int:
        """Compute number of chunks needed to cover some seconds of audio."""
        return int(math.ceil(seconds / (self.chunk_size / self.sample_rate)))

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
    def wav_to_template(wav_file, name: str = "") -> Template:
        """Convert pre-trimmed WAV file to wakeword template."""
        sample_rate, wav_data = scipy.io.wavfile.read(wav_file)
        duration_sec = len(wav_data) / sample_rate
        wav_mfcc = python_speech_features.mfcc(wav_data, sample_rate)

        return Template(name=name, duration_sec=duration_sec, mfcc=wav_mfcc)
