"""Implementation of Snips Personal Wake Word Detector."""
import logging
import math
import typing
from dataclasses import dataclass
from enum import Enum

import numpy as np
import python_speech_features
import scipy.io.wavfile
from dtw import dtw
from rhasspysilence import WebRtcVadRecorder

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


class Raven:
    """
    Wakeword detector based on Snips Personal Wake Word Detector.
    https://medium.com/snips-ai/machine-learning-on-voice-a-gentle-introduction-with-snips-personal-wake-word-detector-133bd6fb568e

    Attributes
    ----------

    templates: List[Template]
        Wake word templates created from pre-trimmed WAV files

    distance_threshold: float = 40

    """

    def __init__(
        self,
        templates: typing.List[Template],
        distance_threshold: float,
        dtw_window_size: int = 5,
        sample_rate: int = 16000,
        chunk_size: int = 960,
        shift_sec: float = 0.05,
        refractory_sec: float = 2.0,
        recorder: typing.Optional[WebRtcVadRecorder] = None,
        debug: bool = False,
    ):
        self.templates = templates
        assert self.templates, "No templates"

        self.distance_threshold = distance_threshold
        self.dtw_window_size = dtw_window_size
        self.chunk_size = chunk_size
        self.shift_sec = shift_sec
        self.sample_rate = sample_rate

        # Assume 16-bit samples
        self.sample_width = 2
        self.chunk_seconds = (self.chunk_size / self.sample_width) / self.sample_rate

        # Use or create silence detector
        self.recorder = recorder or WebRtcVadRecorder()

        # Keep previously-computed distances for debugging
        self.last_distances: typing.List[typing.Optional[float]] = [
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
        self.num_silence_chunks = self.seconds_to_chunks(self.frame_duration_sec / 2)
        self.silence_chunks_left = 0
        self.num_refractory_chunks = self.seconds_to_chunks(refractory_sec)
        self.refractory_chunks_left = 0

        self.debug = debug

    def process_chunk(self, chunk: bytes) -> typing.List[int]:
        """Process a single chunk of raw audio data.

        Must be the same length as self.chunk_size.

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

        is_silence = self.recorder.is_silence(chunk)

        if self.state == RavenState.IN_SILENCE:
            # Only silence so far
            if not is_silence:
                # Start recording and checking audio
                self.audio_buffer = bytes()
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

        if self.state == RavenState.IN_SPEECH:
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
                    return matching_indexes

        return []

    def _process_frame(self, frame: np.ndarray) -> typing.List[int]:
        """Process a single overlapping frame of audio data.

        Returns
        -------

        List of matching template indexes
        """
        matching_indexes: typing.List[int] = []
        frame_mfcc = python_speech_features.mfcc(frame, self.sample_rate)

        for i, template in enumerate(self.templates):
            alignment = dtw(
                frame_mfcc,
                template.mfcc,
                distance_only=True,
                window_type="slantedband",
                window_args={"window_size": self.dtw_window_size},
            )

            distance = alignment.distance / (len(frame_mfcc) + len(template.mfcc))

            if self.debug:
                _LOGGER.debug(
                    "Distance for template %s: %s (raw=%s)",
                    i,
                    distance,
                    alignment.distance,
                )

            self.last_distances[i] = distance

            if distance < self.distance_threshold:
                matching_indexes.append(i)

        return matching_indexes

    def seconds_to_chunks(self, seconds: float) -> int:
        """Compute number of chunks needed to cover some seconds of audio."""
        return int(math.ceil(seconds / (self.chunk_size / self.sample_rate)))

    @staticmethod
    def wav_to_template(wav_file, name: str = "") -> Template:
        """Convert pre-trimmed WAV file to wakeword template."""
        sample_rate, wav_data = scipy.io.wavfile.read(wav_file)
        duration_sec = len(wav_data) / sample_rate
        wav_mfcc = python_speech_features.mfcc(wav_data, sample_rate)

        return Template(name=name, duration_sec=duration_sec, mfcc=wav_mfcc)
