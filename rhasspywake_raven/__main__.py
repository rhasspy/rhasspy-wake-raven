#!/usr/bin/env python3
"""Command-line interface for Rhasspy Raven."""
import argparse
import json
import logging
import sys
import threading
import time
import typing
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

from rhasspysilence import WebRtcVadRecorder
from rhasspysilence.const import SilenceMethod

from . import Raven, Template
from .utils import buffer_to_wav, trim_silence

_LOGGER = logging.getLogger("rhasspy-wake-raven")
_EXIT_NOW = False

# -----------------------------------------------------------------------------


@dataclass
class RavenInstance:
    """Running instance of Raven (one per keyword)."""

    thread: threading.Thread
    raven: Raven
    chunk_queue: "Queue[bytes]"


# -----------------------------------------------------------------------------


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(prog="rhasspy-wake-raven")
    parser.add_argument(
        "--keyword",
        action="append",
        nargs="+",
        default=[],
        help="Directory with WAV templates and settings (setting-name=value)",
    )
    parser.add_argument(
        "--chunk-size",
        default=1920,
        help="Number of bytes to read at a time from standard in (default: 1920)",
    )
    parser.add_argument(
        "--record",
        nargs="+",
        help="Record example templates to a directory, optionally with given name format (e.g., 'my-keyword-{n:02d}.wav')",
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
        default=Raven.DEFAULT_SHIFT_SECONDS,
        help=f"Seconds to shift sliding time window on audio buffer (default: {Raven.DEFAULT_SHIFT_SECONDS})",
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
        default=1,
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
        "--max-chunks-in-queue",
        type=int,
        help="Maximum number of audio chunks waiting for processing before being dropped",
    )
    parser.add_argument(
        "--skip-probability-threshold",
        type=float,
        default=0,
        help="Skip additional template calculations if probability is below this threshold",
    )
    parser.add_argument(
        "--failed-matches-to-refractory",
        type=int,
        help="Number of failed template matches before entering refractory period (default: disabled)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Create silence detector.
    # This can be shared by Raven instances because it's not maintaining state.
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
        record_dir = Path(args.record[0])
        if len(args.record) > 1:
            record_format = args.record[1]
        else:
            record_format = "example-{n:02d}.wav"

        return record_templates(record_dir, record_format, recorder, args)

    assert args.keyword, "--keyword is required"

    # Instances of Raven that will run in separate threads
    ravens: typing.List[RavenInstance] = []

    # Queue for detections. Handled in separate thread.
    output_queue = Queue()

    # Load one or more keywords
    for keyword_settings in args.keyword:
        template_dir = Path(keyword_settings[0])
        wav_paths = list(template_dir.glob("*.wav"))
        if not wav_paths:
            _LOGGER.warning("No WAV files found in %s", template_dir)
            continue

        keyword_name = template_dir.name

        # Load audio templates
        keyword_templates = [
            Raven.wav_to_template(p, name=str(p), shift_sec=args.window_shift_seconds)
            for p in wav_paths
        ]

        raven_args = {
            "templates": keyword_templates,
            "keyword_name": keyword_name,
            "recorder": recorder,
            "probability_threshold": args.probability_threshold,
            "minimum_matches": args.minimum_matches,
            "distance_threshold": args.distance_threshold,
            "refractory_sec": args.refractory_seconds,
            "shift_sec": args.window_shift_seconds,
            "skip_probability_threshold": args.skip_probability_threshold,
            "failed_matches_to_refractory": args.failed_matches_to_refractory,
            "debug": args.debug,
        }

        # Apply settings
        average_templates = args.average_templates
        for setting_str in keyword_settings[1:]:
            setting_name, setting_value = setting_str.strip().split("=", maxsplit=1)
            setting_name = setting_name.lower()

            if setting_name == "name":
                raven_args["keyword_name"] = setting_value
            elif setting_name == "probability-threshold":
                raven_args["probability_threshold"] = float(setting_value)
            elif setting_name == "minimum-matches":
                raven_args["minimum_matches"] = int(setting_value)
            elif setting_name == "average-templates":
                average_templates = setting_value.lower().strip() == "true"

        if average_templates:
            _LOGGER.debug(
                "Averaging %s templates for %s", len(keyword_templates), template_dir
            )
            raven_args["templates"] = [Template.average_templates(keyword_templates)]

        # Create instance of Raven in a separate thread for keyword
        raven = Raven(**raven_args)
        chunk_queue: "Queue[bytes]" = Queue()

        ravens.append(
            RavenInstance(
                thread=threading.Thread(
                    target=detect_thread_proc,
                    args=(chunk_queue, raven, output_queue, args),
                    daemon=True,
                ),
                raven=raven,
                chunk_queue=chunk_queue,
            )
        )

    # Start all threads
    for raven_inst in ravens:
        raven_inst.thread.start()

    output_thread = threading.Thread(
        target=output_thread_proc, args=(output_queue,), daemon=True
    )

    output_thread.start()

    # -------------------------------------------------------------------------

    print("Reading 16-bit 16Khz raw audio from stdin...", file=sys.stderr)

    if args.read_entire_input:
        audio_buffer = FakeStdin(sys.stdin.buffer.read())
    else:
        audio_buffer = sys.stdin.buffer

    try:
        while True:
            # Read raw audio chunk
            chunk = audio_buffer.read(args.chunk_size)
            if not chunk or _EXIT_NOW:
                # Empty chunk
                break

            # Add to all detector threads
            for raven_inst in ravens:
                raven_inst.chunk_queue.put(chunk)

    except KeyboardInterrupt:
        pass
    finally:
        if not args.read_entire_input:
            # Exhaust queues
            _LOGGER.debug("Emptying audio queues...")
            for raven_inst in ravens:
                while not raven_inst.chunk_queue.empty():
                    raven_inst.chunk_queue.get()

        for raven_inst in ravens:
            # Signal thread to quit
            raven_inst.chunk_queue.put(None)
            _LOGGER.debug("Waiting for %s thread...", raven_inst.raven.keyword_name)
            raven_inst.thread.join()

        # Stop output thread
        output_queue.put(None)
        _LOGGER.debug("Waiting for output thread...")
        output_thread.join()


# -----------------------------------------------------------------------------


def detect_thread_proc(chunk_queue, raven, output_queue, args):
    """Template matching in a separate thread."""
    global _EXIT_NOW

    detect_tick = 0
    start_time = time.time()

    while True:

        if args.max_chunks_in_queue is not None:
            # Drop audio chunks to bring queue size back down
            dropped_chunks = 0
            while chunk_queue.qsize() > args.max_chunks_in_queue:
                chunk = chunk_queue.get()
                dropped_chunks += 1

            if dropped_chunks > 0:
                _LOGGER.debug("Dropped %s chunks of audio", dropped_chunks)

        chunk = chunk_queue.get()
        if chunk is None:
            # Empty chunk indicates we should exit
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

                output_queue.put(
                    {
                        "keyword": raven.keyword_name,
                        "template": template.name,
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

                if not args.print_all_matches:
                    # Only print first match
                    break

        # Check if we need to exit
        if (args.exit_count is not None) and (detect_tick >= args.exit_count):
            _EXIT_NOW = True


# -----------------------------------------------------------------------------


def record_templates(
    record_dir: Path,
    name_format: str,
    recorder: WebRtcVadRecorder,
    args: argparse.Namespace,
):
    """Record audio templates."""
    print("Reading 16-bit 16Khz mono audio from stdin...", file=sys.stderr)

    num_templates = 0

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

                template_path = record_dir / name_format.format(n=num_templates)
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


def output_thread_proc(dict_queue):
    """Outputs a line of JSON for each detection."""
    while True:
        output_dict = dict_queue.get()
        if output_dict is None:
            break

        print(json.dumps(output_dict), flush=True, ensure_ascii=False)


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
