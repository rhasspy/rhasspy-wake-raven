#!/usr/bin/env python3
"""Command-line interface for Rhasspy Raven."""
import argparse
import json
import logging
import sys
import time
import typing
from pathlib import Path

from rhasspysilence import WebRtcVadRecorder
from rhasspysilence.const import SilenceMethod

from . import Raven, Template
from .utils import buffer_to_wav, trim_silence

_LOGGER = logging.getLogger("rhasspy-wake-raven")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(prog="rhasspy-wake-raven")
    parser.add_argument(
        "templates", nargs="+", help="Path to WAV file templates or directories"
    )
    parser.add_argument(
        "--record",
        help="Record example templates with given name format (e.g., 'okay-rhasspy-{n:02d}.wav')",
    )
    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=0.5,
        help="Probability above which detection occurs (default: 0.5)",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.22,
        help="Normalized dynamic time warping distance threshold for template matching (default: 0.22)",
    )
    parser.add_argument(
        "--minimum-matches",
        type=int,
        default=1,
        help="Number of templates that must match to produce output (default: 1)",
    )
    parser.add_argument(
        "--refractory-seconds",
        type=float,
        default=2.0,
        help="Seconds before wake word can be activated again (default: 2)",
    )
    parser.add_argument(
        "--print-all-matches",
        action="store_true",
        help="Print JSON for all matching templates instead of just the first one",
    )
    parser.add_argument(
        "--window-shift-seconds",
        type=float,
        default=0.01,
        help="Seconds to shift sliding time window on audio buffer (default: 0.01)",
    )
    parser.add_argument(
        "--dtw-window-size",
        type=int,
        default=5,
        help="Size of band around slanted diagonal during dynamic time warping calculation (default: 5)",
    )
    parser.add_argument(
        "--vad-sensitivity",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Webrtcvad VAD sensitivity (1-3)",
    )
    parser.add_argument(
        "--current-threshold",
        type=float,
        help="Debiased energy threshold of current audio frame",
    )
    parser.add_argument(
        "--max-energy",
        type=float,
        help="Fixed maximum energy for ratio calculation (default: observed)",
    )
    parser.add_argument(
        "--max-current-ratio-threshold",
        type=float,
        help="Threshold of ratio between max energy and current audio frame",
    )
    parser.add_argument(
        "--silence-method",
        choices=[e.value for e in SilenceMethod],
        default=SilenceMethod.VAD_ONLY,
        help="Method for detecting silence",
    )
    parser.add_argument(
        "--average-templates",
        action="store_true",
        help="Average wakeword templates together to reduce number of calculations",
    )
    parser.add_argument(
        "--exit-count",
        type=int,
        help="Exit after some number of detections (default: never)",
    )
    parser.add_argument(
        "--read-entire-input",
        action="store_true",
        help="Read entire audio input at start and exit after processing",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Create silence detector
    recorder = WebRtcVadRecorder(
        vad_mode=args.vad_sensitivity,
        silence_method=args.silence_method,
        current_energy_threshold=args.current_threshold,
        max_energy=args.max_energy,
        max_current_ratio_threshold=args.max_current_ratio_threshold,
        min_seconds=0.5,
        before_seconds=1,
    )

    if args.record:
        # Do recording instead of recognizing
        return record_templates(args.record, recorder, args)

    # Load audio templates
    template_paths: typing.List[Path] = []
    for template_path_str in args.templates:
        template_path = Path(template_path_str)
        if template_path.is_dir():
            # Add all WAV files from directory
            _LOGGER.debug("Adding WAV files from directory %s", template_path)
            template_paths.extend(template_path.glob("*.wav"))
        elif template_path.is_file():
            # Add file directly
            template_paths.append(template_path)

    templates = [Raven.wav_to_template(p, name=str(p)) for p in template_paths]
    if args.average_templates:
        _LOGGER.debug("Averaging %s templates", len(templates))
        templates = [Template.average_templates(templates)]

    # Create Raven object
    raven = Raven(
        templates=templates,
        recorder=recorder,
        probability_threshold=args.probability_threshold,
        minimum_matches=args.minimum_matches,
        distance_threshold=args.distance_threshold,
        refractory_sec=args.refractory_seconds,
        shift_sec=args.window_shift_seconds,
        debug=args.debug,
    )

    print("Reading 16-bit 16Khz raw audio from stdin...", file=sys.stderr)

    if args.read_entire_input:
        audio_buffer = FakeStdin(sys.stdin.buffer.read())
    else:
        audio_buffer = sys.stdin.buffer

    try:
        detect_tick = 0
        start_time = time.time()
        while True:
            # Read raw audio chunk
            chunk = audio_buffer.read(raven.chunk_size)
            if not chunk:
                # Empty chunk
                break

            # Ensure chunk is the right size
            while len(chunk) < raven.chunk_size:
                chunk += audio_buffer.read(raven.chunk_size - len(chunk))
                if not chunk:
                    # Empty chunk
                    break

            # Get matching audio templates (if any)
            matching_indexes = raven.process_chunk(chunk)
            if len(matching_indexes) >= args.minimum_matches:
                detect_time = time.time()
                detect_tick += 1

                # Print results for matching templates
                for template_index in matching_indexes:
                    template = raven.templates[template_index]
                    distance = raven.last_distances[template_index]
                    probability = raven.last_probabilities[template_index]

                    print(
                        json.dumps(
                            {
                                "keyword": template.name,
                                "detect_seconds": detect_time - start_time,
                                "detect_timestamp": detect_time,
                                "raven": {
                                    "probability": probability,
                                    "distance": distance,
                                    "probability_threshold": args.probability_threshold,
                                    "distance_threshold": raven.distance_threshold,
                                    "tick": detect_tick,
                                    "matches": len(matching_indexes),
                                    "match_seconds": raven.match_seconds,
                                },
                            }
                        )
                    )

                    if not args.print_all_matches:
                        # Only print first match
                        break

            # Check if we need to exit
            if (args.exit_count is not None) and (detect_tick >= args.exit_count):
                break

    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------


def record_templates(
    name_format: str, recorder: WebRtcVadRecorder, args: argparse.Namespace
):
    """Record audio templates."""
    print("Reading 16-bit 16Khz mono audio from stdin...", file=sys.stderr)

    num_templates = 0
    template_dir = Path(args.templates[0])

    try:
        print(
            f"Recording template {num_templates}. Please speak your wake word. Press CTRL+C to exit."
        )
        recorder.start()

        while True:
            # Read raw audio chunk
            chunk = sys.stdin.buffer.read(recorder.chunk_size)
            if not chunk:
                # Empty chunk
                break

            result = recorder.process_chunk(chunk)
            if result:
                audio_bytes = recorder.stop()
                audio_bytes = trim_silence(audio_bytes)

                template_path = template_dir / name_format.format(n=num_templates)
                template_path.parent.mkdir(parents=True, exist_ok=True)

                wav_bytes = buffer_to_wav(audio_bytes)
                template_path.write_bytes(wav_bytes)
                _LOGGER.debug(
                    "Wrote %s byte(s) of WAV audio to %s", len(wav_bytes), template_path
                )

                num_templates += 1
                print(
                    f"Recording template {num_templates}. Please speak your wake word. Press CTRL+C to exit."
                )
                recorder.start()
    except KeyboardInterrupt:
        print("Done")


# -----------------------------------------------------------------------------


class FakeStdin:
    """Wrapper for fixed audio buffer that returns empty chunks when exhausted."""

    def __init__(self, audio_bytes: bytes):
        self.audio_bytes = audio_bytes

    def read(self, n: int) -> bytes:
        """Read n bytes from buffer or return empty chunk."""
        if len(self.audio_bytes) >= n:
            chunk = self.audio_bytes[:n]
            self.audio_bytes = self.audio_bytes[n:]
            return chunk

        # Empty chunk
        return bytes()


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
