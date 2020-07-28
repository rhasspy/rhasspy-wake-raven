# Rhasspy Raven Wakeword System

Wakeword detector based on the [Snips Personal Wake Word Detector](https://medium.com/snips-ai/machine-learning-on-voice-a-gentle-introduction-with-snips-personal-wake-word-detector-133bd6fb568e).

The underlying implementation of Raven heavily borrows from [node-personal-wakeword](https://github.com/mathquis/node-personal-wakeword) by [mathquis](https://github.com/mathquis).

## Dependencies

* Python 3.7
* `python-speech-features` for [MFCC](https://python-speech-features.readthedocs.io/en/latest/) computation
* `rhasspy-silence` for [silence detection](https://github.com/rhasspy/rhasspy-silence)
* Scientific libraries
    * `sudo apt-get install liblapack3 libatlas-base-dev`

## Installation

```sh
$ git clone https://github.com/rhasspy/rhasspy-wake-raven.git
$ cd rhasspy-wake-raven
$ ./configure
$ make
$ make install
```

## Recording Templates

Record at least 3 WAV templates with your wake word:

```sh
$ arecord -r 16000 -f S16_LE -c 1 -t raw | \
    bin/rhasspy-wake-raven --record keyword-dir/
```

Follow the prompts and speak your wake word. When you've recorded at least 3 examples, hit CTRL+C to exit. Your WAV templates will have silence automatically trimmed, and will be saved in the directory `keyword-dir/`. Add a format string after the directory name to control the file names:

```sh
$ arecord -r 16000 -f S16_LE -c 1 -t raw | \
    bin/rhasspy-wake-raven --record keyword-dir/ 'keyword-{n:02d}.wav'
```

The format string will receive the 0-based index `n` for each example.

If you want to manually record WAV templates, trim silence off the front and back and make sure to export them as 16-bit 16Khz mono WAV files.

## Running

After recording your WAV templates in a directory, run:

```sh
$ arecord -r 16000 -f S16_LE -c 1 -t raw | \
    bin/rhasspy-wake-raven --keyword <WAV_DIR> ...
```

where `<WAV_DIR>` contains the WAV templates. You may add as many keywords as you'd like, though this will use additional CPU. It's recommended you use `--average-templates` to keep CPU usage down.

Some settings can be specified per-keyword:

```sh
$ arecord -r 16000 -f S16_LE -c 1 -t raw | \
    bin/rhasspy-wake-raven \
      --keyword keyword1/ name=my-keyword1 probability-threshold=0.45 minimum-matches=2 \
      --keyword keyword2/ name=my-keyword2 probability-threshold=0.55 average-templates=true
```

If not set, `probability-threshold=`, etc. fall back on the values supplied to `--probability-threshold`, etc.

Add `--debug` to the command line to get more information about the underlying computation on each audio frame.

### Example

Using the example files for "okay rhasspy":

```sh
$ arecord -r 16000 -f S16_LE -c 1 -t raw | \
    bin/rhasspy-wake-raven --keyword etc/okay-rhasspy/
```

This requires at least 1 of the 3 WAV templates to match before output like this is printed:

```json
{"keyword": "okay-rhasspy", "template": "etc/okay-rhasspy/okay-rhasspy-00.wav", "detect_seconds": 2.7488508224487305, "detect_timestamp": 1594996988.638912, "raven": {"probability": 0.45637207995699963, "distance": 0.25849045215799454, "probability_threshold": 0.5, "distance_threshold": 0.22, "tick": 1, "matches": 2, "match_seconds": 0.005367016012314707}}
```

Use `--minimum-matches` to change how many templates must match for a detection to occur or `--average-templates` to combine all WAV templates into a single template (reduces CPU usage). Adjust the sensitivity with `--probability-threshold` which sets the lower bound of the detection probability (default is 0.5).

## Output

Raven outputs a line of JSON when the wake word is detected. Fields are:

* `keyword` - name of keyword or directory
* `template` - path to WAV file template
* `detect_seconds` - seconds after start of program when detection occurred
* `detect_timestamp` - timestamp when detection occurred (using `time.time()`)
* `raven`
    * `probability` - detection probability
    * `probability_threshold` - range of probabilities for detection
    * `distance` - normalized dynamic time warping distance
    * `distance_threshold` - distance threshold used for comparison
    * `matches` - number of WAV templates that matched
    * `match_seconds` - seconds taken for dynamic time warping calculations
    * `tick` - monotonic counter incremented for each detection

## Testing

You can test how well Raven works on a set of sample WAV files:

```sh
$ PATH=$PWD/bin:$PATH test-raven.py --test-directory /path/to/samples/ --keyword /path/to/templates/
```

This will run up to 10 parallel instances of Raven (change with `--test-workers`) and output a JSON report with detection information and summary statistics like:

```json
{
  "positive": [...],
  "negative": [...],
  "summary": {
    "true_positives": 14,
    "false_positives": 0,
    "true_negatives": 40,
    "false_negatives": 7,
    "precision": 1.0,
    "recall": 0.6666666666666666,
    "f1_score": 0.8
}
```

Any additional command-line arguments are passed to Raven (e.g., `--minimum-matches`).

## Command-Line Interface

```
usage: rhasspy-wake-raven [-h] [--keyword KEYWORD [KEYWORD ...]]
                          [--chunk-size CHUNK_SIZE]
                          [--record RECORD [RECORD ...]]
                          [--probability-threshold PROBABILITY_THRESHOLD]
                          [--distance-threshold DISTANCE_THRESHOLD]
                          [--minimum-matches MINIMUM_MATCHES]
                          [--refractory-seconds REFRACTORY_SECONDS]
                          [--print-all-matches]
                          [--window-shift-seconds WINDOW_SHIFT_SECONDS]
                          [--dtw-window-size DTW_WINDOW_SIZE]
                          [--vad-sensitivity {1,2,3}]
                          [--current-threshold CURRENT_THRESHOLD]
                          [--max-energy MAX_ENERGY]
                          [--max-current-ratio-threshold MAX_CURRENT_RATIO_THRESHOLD]
                          [--silence-method {vad_only,ratio_only,current_only,vad_and_ratio,vad_and_current,all}]
                          [--average-templates] [--exit-count EXIT_COUNT]
                          [--read-entire-input]
                          [--max-chunks-in-queue MAX_CHUNKS_IN_QUEUE]
                          [--skip-probability-threshold SKIP_PROBABILITY_THRESHOLD]
                          [--failed-matches-to-refractory FAILED_MATCHES_TO_REFRACTORY]
                          [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --keyword KEYWORD [KEYWORD ...]
                        Directory with WAV templates and settings (setting-
                        name=value)
  --chunk-size CHUNK_SIZE
                        Number of bytes to read at a time from standard in
                        (default: 1920)
  --record RECORD [RECORD ...]
                        Record example templates to a directory, optionally
                        with given name format (e.g., 'my-
                        keyword-{n:02d}.wav')
  --probability-threshold PROBABILITY_THRESHOLD
                        Probability above which detection occurs (default:
                        0.5)
  --distance-threshold DISTANCE_THRESHOLD
                        Normalized dynamic time warping distance threshold for
                        template matching (default: 0.22)
  --minimum-matches MINIMUM_MATCHES
                        Number of templates that must match to produce output
                        (default: 1)
  --refractory-seconds REFRACTORY_SECONDS
                        Seconds before wake word can be activated again
                        (default: 2)
  --print-all-matches   Print JSON for all matching templates instead of just
                        the first one
  --window-shift-seconds WINDOW_SHIFT_SECONDS
                        Seconds to shift sliding time window on audio buffer
                        (default: 0.02)
  --dtw-window-size DTW_WINDOW_SIZE
                        Size of band around slanted diagonal during dynamic
                        time warping calculation (default: 5)
  --vad-sensitivity {1,2,3}
                        Webrtcvad VAD sensitivity (1-3)
  --current-threshold CURRENT_THRESHOLD
                        Debiased energy threshold of current audio frame
  --max-energy MAX_ENERGY
                        Fixed maximum energy for ratio calculation (default:
                        observed)
  --max-current-ratio-threshold MAX_CURRENT_RATIO_THRESHOLD
                        Threshold of ratio between max energy and current
                        audio frame
  --silence-method {vad_only,ratio_only,current_only,vad_and_ratio,vad_and_current,all}
                        Method for detecting silence
  --average-templates   Average wakeword templates together to reduce number
                        of calculations
  --exit-count EXIT_COUNT
                        Exit after some number of detections (default: never)
  --read-entire-input   Read entire audio input at start and exit after
                        processing
  --max-chunks-in-queue MAX_CHUNKS_IN_QUEUE
                        Maximum number of audio chunks waiting for processing
                        before being dropped
  --skip-probability-threshold SKIP_PROBABILITY_THRESHOLD
                        Skip additional template calculations if probability
                        is below this threshold
  --failed-matches-to-refractory FAILED_MATCHES_TO_REFRACTORY
                        Number of failed template matches before entering
                        refractory period (default: disabled)
  --debug               Print DEBUG messages to the console
```
