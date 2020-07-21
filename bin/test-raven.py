#!/usr/bin/env python3
"""Test WAV files using rhasspy-wake-raven.

Uses the same test directory structure as Mycroft Precise.
WAV files should be in:

test/
  wake-word/
    positive-example-00.wav
    positive-example-01.wav
    ...
  not-wake-word/
    negative-example-00.wav
    negative-example-01.wav
    ...

Outputs a JSON report with detection info and statistics.
"""
import argparse
import concurrent.futures
import io
import json
import logging
import subprocess
import sys
import typing
import wave
from pathlib import Path

_LOGGER = logging.getLogger("rhasspy-wake-raven")

# Two seconds of silence assuming 16-bit 16Khz mono
_SILENCE = bytes(2 * 16000 * 2)

# -----------------------------------------------------------------------------


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(prog="test-raven.py")
    parser.add_argument(
        "--test-directory",
        required=True,
        help="Path to directory with wake-word/not-wake-word",
    )
    parser.add_argument(
        "--test-workers",
        type=int,
        default=10,
        help="Number of simultaneous raven processes during test (default: 10)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args, raven_args = parser.parse_known_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        raven_args.append("--debug")
    else:
        logging.basicConfig(level=logging.INFO)

    # Raven will exit after 1 detection
    raven_args.extend(["--exit-count", "1", "--read-entire-input"])

    args.test_directory = Path(args.test_directory)
    wake_word_dir = args.test_directory / "wake-word"
    not_wake_word_dir = args.test_directory / "not-wake-word"

    assert (
        wake_word_dir.is_dir() or not_wake_word_dir.is_dir()
    ), f"Expected either wake-word or not-wake-word directory in {args.test_directory}"

    results = {
        "positive": [
            {"wav_path": str(p.absolute())} for p in list(wake_word_dir.rglob("*.wav"))
        ],
        "negative": [
            {"wav_path": str(p.absolute())}
            for p in list(not_wake_word_dir.rglob("*.wav"))
        ],
    }

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # Test positive and negative examples in parallel
    path_to_example: typing.Dict[str, typing.Dict[str, typing.Any]] = {}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.test_workers
    ) as executor:
        # future -> (is_positive, example)
        future_to_example = {}

        for pos_example in results["positive"]:
            wav_path = pos_example["wav_path"]
            path_to_example[wav_path] = pos_example
            future = executor.submit(raven, wav_path, raven_args)

            future_to_example[future] = (True, wav_path)

        for neg_example in results["negative"]:
            wav_path = neg_example["wav_path"]
            path_to_example[wav_path] = neg_example
            future = executor.submit(raven, wav_path, raven_args)

            future_to_example[future] = (False, wav_path)

        for future in concurrent.futures.as_completed(future_to_example):
            is_positive, wav_path = future_to_example[future]
            example = path_to_example[wav_path]
            detection = future.result()

            if detection:
                if is_positive:
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                if is_positive:
                    false_negatives += 1
                else:
                    true_negatives += 1

            example["detection"] = detection

    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0.0

    try:
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0.0

    try:
        f1_score = 2 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        f1_score = 0.0

    results["summary"] = {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }

    json.dump(results, sys.stdout, indent=4)


# -----------------------------------------------------------------------------


def raven(wav_path: str, raven_args: typing.List[str]) -> typing.Dict[str, typing.Any]:
    """Runs Rhasspy raven on a single WAV file."""
    raven_command = ["rhasspy-wake-raven"] + raven_args
    _LOGGER.debug(raven_command)
    raven_proc = subprocess.Popen(
        raven_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )

    with open(wav_path, "rb") as wav_file:
        wav_bytes = wav_file.read()

    wav_data = _SILENCE + wav_to_buffer(wav_bytes) + _SILENCE

    stdout, stderr = raven_proc.communicate(input=wav_data)
    raven_proc.terminate()
    raven_proc.wait()
    assert raven_proc.returncode == 0, stderr

    if stdout:
        return json.loads(stdout)

    return {}


# -----------------------------------------------------------------------------


def wav_to_buffer(wav_bytes: bytes) -> bytes:
    """Return the raw audio of a WAV file"""
    with io.BytesIO(wav_bytes) as wav_buffer:
        wav_file: wave.Wave_read = wave.open(wav_buffer, "rb")
        with wav_file:
            frames = wav_file.getnframes()
            return wav_file.readframes(frames)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
