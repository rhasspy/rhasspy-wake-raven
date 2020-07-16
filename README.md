# Rhasspy Raven Wakeword System

Wakeword detector based on the [Snips Personal Wake Word Detector](https://medium.com/snips-ai/machine-learning-on-voice-a-gentle-introduction-with-snips-personal-wake-word-detector-133bd6fb568e).

## Dependencies

* Python 3.7
* `dtw-python` for [Dynamic Time Warping](https://dynamictimewarping.github.io/python/) calculation
* `python-speech-features` for [MFCC](https://python-speech-features.readthedocs.io/en/latest/) computation
* `rhasspy-silence` for [silence detection](https://github.com/rhasspy/rhasspy-silence)

## Installation

```sh
$ git clone https://github.com/rhasspy/rhasspy-wake-raven.git
$ cd rhasspy-wake-raven
$ ./configure
$ make
$ make install
```

## Running

Record at least 3 WAV files with your wake word. Trim silence off the front and back manually, and export them as 16-bit 16Khz mono WAV files. Then, run:

```sh
$ arecord -r 16000 -f S16_LE -c 1 -t raw | \
    bin/rhasspy-wake-raven --distance-threshold D <WAV1> <WAV2> <WAV3> ...
```

where `D` is a threshold used to determine if a WAV template matches. A value of 38 for `D` seemed to work for the sample WAV files in `etc/test`. You can add `--debug` to the command line to see the distance values for each match attempt to get an idea for your wakeword.

### Example

Using the example files for "okay rhasspy":

```sh
$ arecord -r 16000 -f S16_LE -c 1 -t raw | \
    bin/rhasspy-wake-raven --distance-threshold 38 --minimum-matches 2 etc/test/okay-rhasspy-*.wav
```

This requires at least 2 of the 3 WAV templates to match before output is printed:

```json
{"keyword": "etc/test/okay-rhasspy-00.wav", "detect_seconds": 5.871257066726685, "detect_timestamp": 1594929004.2440836, "raven": {"distance": 35.00761344404474, "threshold": 38.0, "tick": 1, "matches": 3}}
```

## Output

Raven outputs a line of JSON when the wake word is detected. Fields are:

* `keyword` - path to WAV file template
* `detect_seconds` - seconds after start of program when detection occurred
* `detect_timestamp` - timestamp when detection occurred (using `time.time()`)
* `raven`
    * `distance` - normalized dynamic time warping distance
    * `threshold` - threshold used for comparison
    * `matches` - number of WAV templates that matched
    * `tick` - monotonic counter incremented for each detection
