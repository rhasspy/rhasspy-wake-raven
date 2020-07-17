#!/usr/bin/env python3
"""Command-line interface for Rhasspy Raven."""
import argparse
import json
import logging
import sys
import time

from rhasspysilence import WebRtcVadRecorder
from rhasspysilence.const import SilenceMethod

from . import Raven, Template

_LOGGER = logging.getLogger("rhasspy-wake-raven")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(prog="rhasspy-wake-raven")
    parser.add_argument("templates", nargs="+", help="Path to WAV file templates")
    parser.add_argument(
        "--probability-threshold",
        type=float,
        nargs=2,
        default=[0.45, 0.55],
        help="Probability range where detection occurs (default: (0.45, 0.55))",
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
        default=0.05,
        help="Seconds to shift sliding time window on audio buffer (default: 0.05)",
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
    )

    # Load audio templates
    templates = [Raven.wav_to_template(p, name=p) for p in args.templates]
    if args.average_templates:
        _LOGGER.debug("Averaging %s templates", len(templates))
        templates = [Template.average_templates(templates)]

    # Create Raven object
    raven = Raven(
        templates=templates,
        recorder=recorder,
        probability_threshold=tuple(args.probability_threshold),
        distance_threshold=args.distance_threshold,
        refractory_sec=args.refractory_seconds,
        shift_sec=args.window_shift_seconds,
        debug=args.debug,
    )

    print("Reading 16-bit 16Khz raw audio from stdin...", file=sys.stderr)

    try:
        detect_tick = 0
        start_time = time.time()
        while True:
            # Read raw audio chunk
            chunk = sys.stdin.buffer.read(raven.chunk_size)

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
                                },
                            }
                        )
                    )

                    if not args.print_all_matches:
                        # Only print first match
                        break

    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
