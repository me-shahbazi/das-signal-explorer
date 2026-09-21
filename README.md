# FiberHome DAS offline tools

This repository contains Python tools for reading and analysing FiberHome
Distributed Acoustic Sensing (DAS) recordings saved as .dat files. It also
includes a Windows voice player that turns the time trace at one fibre
distance into audible sound.

The raw captures are deliberately **not** included in this repository.

## Contents

| Path | Purpose |
| --- | --- |
| scripts/parseDataHeader.py | Parses the 256-byte little-endian DAS header. |
| scripts/DASinterogator_offline.py | Loads a distance region of interest, removes DC, and provides waterfall, frequency-distance, and event-detection helpers. |
| scripts/VoicePlayer.py | Plays the time trace at one selected fibre distance. |
| fiberhome_realtime_viewer.py | Headless watcher and per-file processing pipeline. |
| data_180_240/Params.json | Acquisition metadata for the main example recording. |
| data_SixtySix/Params.json | Acquisition metadata for a second example recording. |
| requirements.txt | Tested Python package versions. |

## Data format

Each DAS file is structured as follows:

- A 256-byte header containing acquisition metadata.
- A payload of signed int16 samples arranged as
  (frames_per_file, num_traces).

The header reader identifies the channel, spatial resolution, fibre length,
number of spatial traces, PRF / scan rate, and frames per file. The PRF is the
time-sampling rate for a fixed distance trace. It is read from each file
header; it is not inferred from sample_rate_MHz.

The current data_180_240 capture has 202 channel-1 files with these settings:

| Setting | Value |
| --- | --- |
| Fibre length | 2000 m |
| Spatial traces | 5000 |
| Spatial resolution | 0.4 m |
| PRF | 1500 Hz |
| Frames per file | 1024 |
| Total duration | about 137.9 s |

The corresponding scenario is stepping at about 180 m, moving to 240 m, then
returning to 180 m.

## Installation

The project was tested with Python 3.14.3 on Windows. The voice player uses
the Windows standard-library module winsound; no extra playback package is
needed.

From the project root in Command Prompt:

~~~bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
~~~

In PowerShell, activate the same environment with:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

Then create the generated-results directory:

~~~powershell
New-Item -ItemType Directory -Force output
~~~

The output directory is intentionally ignored by Git because scripts can
write plots and CSV results there. It must exist before using the current
VoicePlayer.py.

## Providing raw recordings

Create or copy the local data folders beside scripts/:

~~~text
rawData/
├── data_180_240/
│   ├── Params.json
│   └── ch10__*.dat
├── data_SixtySix/
│   ├── Params.json
│   └── ch10__*.dat
└── scripts/
~~~

Raw .dat files, saved NumPy arrays, videos, WAV files, generated outputs,
and .venv are excluded by .gitignore.

## Voice playback

VoicePlayer.py reads the selected point through DASinterogator_offline,
removes DC, optionally applies a gentle high-pass filter, resamples to a
standard Windows playback rate, and plays a mono WAV from memory. It does not
save an audio file.

Run it from the project root:

~~~powershell
python scripts/VoicePlayer.py
~~~

The default is the complete data_180_240 recording at 100 m. Playback is at
the original time scale. The program reports the actual spatial bin selected,
the PRF read from the header, the selected time range, and the applied gain.

Examples:

~~~powershell
# Play five seconds at 188 m, beginning ten seconds into the recording.
python scripts/VoicePlayer.py --distance 188 --start 10 --duration 5

# Play without the additional 10 Hz high-pass filter.
python scripts/VoicePlayer.py --distance 188 --no-filter

# Use a different local capture folder.
python scripts/VoicePlayer.py --data-dir data_SixtySix

# Use a fixed gain when comparing the relative level of several points.
python scripts/VoicePlayer.py --distance 188 --gain 350
~~~

Without --gain, the player uses automatic peak normalization independently
for each selected point or interval. This is usually the best choice for
listening. With --gain, the supplied fixed gain is multiplied by
--volume (whose default is 1.0). Use the same --gain, --volume, and filter
settings when comparing levels between locations. A fixed gain that would
clip the audio is rejected instead of silently distorting it.

The audio represents the sampled DAS trace, not a calibrated microphone
recording. The highest recoverable frequency is approximately half the PRF:
750 Hz for the current 1500 Hz capture, or 5 kHz for a 10 kHz capture.

VoicePlayer.py currently uses synchronous winsound memory playback. The
terminal waits until playback ends; Ctrl+C may not stop the current sound
immediately.

## Offline analysis

DASinterogator_offline is intended for an interactive Python session. A
typical workflow is:

~~~python
from scripts.DASinterogator_offline import DASinterogator_offline

das = DASinterogator_offline(
    baseDir=".",
    dataDir="data_180_240",
    startDist=175,
    endDist=250,
)

data_roi, time_axis, distance_axis = das.readPeriod(period_s=30)
das.displayWaterFall(data_roi, time_axis, distance_axis, SaveMode=True)

fft_db, frequencies, _ = das._FFTcalculator(data_roi)
das.displayFreqvsDist(fft_db, frequencies, distance_axis, SaveMode=True)
~~~

The analysis methods use the PRF and spatial configuration parsed from the
first file header. Keep files with different acquisition settings in separate
folders.

## GitHub

Before committing, verify that raw recordings and local build artifacts are
not staged:

~~~powershell
git status
git diff --cached --name-only
~~~

Expected tracked material is source code, Params.json, .gitignore,
requirements.txt, and this README. Do not add .venv/, *.dat, output/, or
generated media.
