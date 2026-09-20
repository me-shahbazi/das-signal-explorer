"""Play one DAS distance trace through the Windows default audio device.

Examples (from the rawData project root):
    from scripts.VoicePlayer import DASVoicePlayer
    player = DASVoicePlayer()
    player.play()  # 100 m, all data_180_240 files, actual acquisition speed
    player.play(distance_m=180, start_s=10, duration_s=5)
    player.play(highpass_hz=None)  # no additional high-pass filter
    player.play(gain=20.0)  # optional fixed gain; use the same gain to compare points

Run this file directly to play the defaults; use --help for other options.
Requires the project's existing NumPy and SciPy installations. Playback uses
Windows winsound, with a mono WAV held in memory; no audio file is saved.

The signed time-domain trace is treated as the measurement to sonify. No I/Q
extraction, phase unwrapping, or claim of calibrated acoustic pressure is made.
Time is cumulative sample time across sorted files, as in the existing reader;
filename timestamp jitter does not stretch playback or insert silence.
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path
import sys
import wave

# Direct execution should not create imported modules' bytecode files.
if __name__ == "__main__":
    sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from scripts.DASinterogator_offline import DASinterogator_offline
from scripts.parseDataHeader import parse_das_header


class DASVoicePlayer:
    """Offline, blocking mono playback using DASinterogator_offline.

    PRF always comes from the headers (1500 Hz in the current capture, or
    10000 Hz in a future capture with that PRF). It is never inferred from
    the ADC sample_rate_MHz or silently replaced by a default.

    Only play() is a public operation. Relative data paths are resolved
    against rawData/, independently of the current working directory.
    """

    def __init__(self, data_dir=None):
        self.base_dir = ROOT
        directory = Path(data_dir) if data_dir is not None else Path("data_180_240")
        self.data_dir = (
            directory if directory.is_absolute() else self.base_dir / directory
        ).resolve()

    @staticmethod
    def _finite(value, name):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite number.") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must be a finite number.")
        return number

    def _inspect_files(self):
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data folder does not exist: {self.data_dir}")
        paths = sorted(self.data_dir.glob("ch10__*.dat"))
        if not paths:
            raise FileNotFoundError(f"No channel-1 DAT files in {self.data_dir}")

        fields = (
            "channel_ID", "sample_rate_MHz", "pulse_width",
            "prf_or_scan_rate_Hz", "num_traces", "frames_per_file",
        )
        reference = None
        for path in paths:
            try:
                with path.open("rb") as stream:
                    header = parse_das_header(stream.read(256))
            except (OSError, ValueError, TypeError, ZeroDivisionError) as exc:
                raise ValueError(f"Cannot read a valid DAS header: {path.name}") from exc

            for field in (
                "prf_or_scan_rate_Hz", "frames_per_file", "num_traces",
                "sample_rate_MHz", "Dist_accuracy_m",
            ):
                if self._finite(header.get(field), field) <= 0:
                    raise ValueError(f"{path.name}: invalid {field}.")
            if header["channel_ID"] != 1:
                raise ValueError(f"{path.name}: expected channel 1.")

            if reference is None:
                reference = header
            elif any(header[field] != reference[field] for field in fields):
                raise ValueError(
                    f"Acquisition settings change at {path.name}. "
                    "Use a folder with consistent channel, PRF and spatial settings."
                )

            expected_size = 256 + 2 * header["frames_per_file"] * header["num_traces"]
            if path.stat().st_size != expected_size:
                raise ValueError(
                    f"{path.name}: payload size does not match its header "
                    f"(expected {expected_size} bytes)."
                )
        # paths is known to be non-empty above, so a valid header must have
        # been assigned; make that invariant explicit for static type checkers.
        if reference is None:
            raise ValueError(f"No valid DAS headers found in {self.data_dir}")
        return paths, reference

    def _read_point(self, paths, header, distance_m, first_sample, stop_sample):
        spacing = float(header["Dist_accuracy_m"])
        trace_count = int(header["num_traces"])
        index = min(trace_count - 1, int(math.floor(distance_m / spacing + 0.5)))
        actual_distance = index * spacing

        # The existing constructor calls makedirs(output). Require the project's
        # existing directory so invoking the player cannot create that folder.
        if not (self.base_dir / "output").is_dir():
            raise FileNotFoundError(
                "The existing DAS reader expects rawData/output to exist. "
                "VoicePlayer will not create this directory."
            )
        reader = DASinterogator_offline(
            baseDir=str(self.base_dir),
            dataDir=str(self.data_dir),
            startDist=actual_distance,
            endDist=actual_distance,
        )
        # Preserve an exact single-bin ROI even at floating-point boundaries.
        reader.min_idx = index
        reader.max_idx = index + 1

        frames = int(header["frames_per_file"])
        source_rate = int(header["prf_or_scan_rate_Hz"])
        first_file = first_sample // frames
        stop_file = (stop_sample + frames - 1) // frames
        selected = paths[first_file:stop_file]
        reader.paths = [str(path) for path in selected]

        # readPeriod uses int(1 + period_s * PRF / frames). A half-block margin
        # requests exactly len(selected) blocks, including for a complete folder,
        # without allocating an extra unfilled block before DC subtraction.
        request_s = (len(selected) - 0.5) * frames / source_rate
        data, _, _ = reader.readPeriod(period_s=request_s) # type: ignore
        if data.shape != (len(selected) * frames, 1):
            raise RuntimeError("DAS reader returned an unexpected number of samples.")

        offset = first_file * frames
        # Keep full boundary files as context for filtering before time cropping.
        return (
            data[:, 0].astype(np.float64),
            first_sample - offset,
            stop_sample - offset,
            actual_distance,
        )

    @staticmethod
    def _prepare_audio(trace, source_rate, first, stop, highpass_hz, volume, fixed_gain=None):
        if not np.all(np.isfinite(trace)):
            raise ValueError("DAS trace contains non-finite samples.")

        if highpass_hz is not None:
            # A low-order high-pass attenuates slow drift, not a speech band.
            # Forward/backward filtering avoids a phase delay. Its gain near
            # and below the cutoff is reduced; highpass_hz=None bypasses it.
            sos = butter(2, highpass_hz, btype="highpass", fs=source_rate, output="sos")
            trace = sosfiltfilt(sos, trace, padlen=min(9, trace.size - 2))

        audio = trace[first:stop].copy()
        if highpass_hz is None:
            # readPeriod removed the full-file mean; also center this time crop.
            audio -= np.mean(audio)

        # Select a standard playback rate, never below the acquisition rate.
        # Resampling changes sample count, not elapsed time or pitch.
        output_rate = next(
            (rate for rate in (44100, 48000, 96000) if rate >= source_rate), None
        )
        if output_rate is None:
            raise ValueError("PRF is above the supported playback sample rates.")
        divisor = math.gcd(source_rate, output_rate)
        audio = resample_poly(
            audio, output_rate // divisor, source_rate // divisor, padtype="line"
        )
        target_count = round((stop - first) * output_rate / source_rate)
        audio = audio[:target_count]

        # Only the two playback ends fade (5 ms); file joins are not faded.
        # This suppresses an artificial click when starting/stopping a time crop.
        fade_count = min(round(0.005 * output_rate), audio.size // 2)
        if fade_count >= 2:
            ramp = np.linspace(0.0, 1.0, fade_count)
            audio[:fade_count] *= ramp
            audio[-fade_count:] *= ramp[::-1]

        # Default: preserve the existing automatic peak normalization exactly.
        # Fixed mode: volume is a multiplier; do not normalize each selection.
        peak = float(np.max(np.abs(audio)))
        if fixed_gain is None:
            gain = volume / peak if peak > 1e-12 else 0.0
        else:
            gain = fixed_gain * volume
            if not math.isfinite(gain) or peak * gain > 1.0:
                raise ValueError(
                    "Fixed gain would clip the audio. Lower --gain or --volume; "
                    "automatic normalization is disabled in fixed-gain mode."
                )
        pcm = np.rint(np.clip(audio * gain, -1.0, 1.0) * 32767).astype("<i2")
        return pcm, output_rate, gain

    @staticmethod
    def _play_pcm(pcm, output_rate):
        try:
            import winsound
        except ImportError as exc:
            raise RuntimeError("This playback backend requires Windows.") from exc

        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(output_rate)
                wav.writeframes(pcm.tobytes())
            try:
                # Without SND_ASYNC, memory playback waits until the sound ends.
                winsound.PlaySound(
                    buffer.getvalue(), winsound.SND_MEMORY | winsound.SND_NODEFAULT
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "Windows could not play the audio. Check the default output device."
                ) from exc

    def play(
        self,
        distance_m=100.0,
        start_s=0.0,
        duration_s=None,
        *,
        highpass_hz: float | None = 10.0,
        volume=0.8,
        gain: float | None = None,
    ):
        """Read, prepare and play a distance trace; return playback information.

        distance_m: nearest spatial bin, within the stored bin-center range.
        start_s: seconds from the first sample; rounded to the nearest sample.
        duration_s: None means all remaining samples; otherwise a positive
            duration, clipped to the available recording. No silence is appended.
        highpass_hz: gentle high-pass cutoff, or None for DC removal only.
            Resampling still uses its necessary interpolation filter.
        volume: peak output level in [0, 1] in automatic mode; an amplitude
            multiplier in fixed-gain mode. Independent of Windows volume.
        gain: None preserves automatic peak normalization. A nonnegative fixed
            multiplier is applied to the reader's normalized signal after
            filtering/resampling, with effective gain = gain * volume.
            Use equal gain, volume and filter settings to compare points.
            Excessive fixed gain raises an error rather than clipping or
            silently renormalizing the signal.

        A selection must contain at least two source samples. Processing occurs
        before synchronous playback; importing this module does not play sound.
        """
        distance_m = self._finite(distance_m, "distance_m")
        start_s = self._finite(start_s, "start_s")
        volume = self._finite(volume, "volume")
        if start_s < 0:
            raise ValueError("start_s must be >= 0.")
        if not 0 <= volume <= 1:
            raise ValueError("volume must be between 0 and 1.")
        if gain is not None:
            gain = self._finite(gain, "gain")
            if gain < 0:
                raise ValueError("gain must be >= 0.")
        if duration_s is not None:
            duration_s = self._finite(duration_s, "duration_s")
            if duration_s <= 0:
                raise ValueError("duration_s must be > 0.")

        paths, header = self._inspect_files()
        source_rate = int(header["prf_or_scan_rate_Hz"])
        total_samples = len(paths) * int(header["frames_per_file"])
        total_duration = total_samples / source_rate
        last_distance = (header["num_traces"] - 1) * header["Dist_accuracy_m"]
        if not 0 <= distance_m <= last_distance:
            raise ValueError(f"distance_m must be between 0 and {last_distance:g} m.")
        if start_s >= total_duration:
            raise ValueError(f"start_s must be less than {total_duration:.6f} s.")
        if highpass_hz is not None:
            highpass_hz = self._finite(highpass_hz, "highpass_hz")
            if not 0 < highpass_hz < source_rate / 2:
                raise ValueError(f"highpass_hz must be between 0 and {source_rate / 2:g} Hz.")

        first = round(start_s * source_rate)
        stop = total_samples
        if duration_s is not None:
            end_s = min(total_duration, start_s + duration_s)
            stop = min(total_samples, round(end_s * source_rate))
        if stop - first < 2:
            raise ValueError("The selected interval contains fewer than two samples.")

        trace, crop_start, crop_stop, actual_distance = self._read_point(
            paths, header, distance_m, first, stop
        )
        pcm, output_rate, applied_gain = self._prepare_audio(
            trace, source_rate, crop_start, crop_stop, highpass_hz, volume,
            fixed_gain=gain,
        )
        info = {
            "requested_distance_m": distance_m,
            "actual_distance_m": actual_distance,
            "prf_hz": source_rate,
            "output_sample_rate_hz": output_rate,
            "start_s": first / source_rate,
            "duration_s": (stop - first) / source_rate,
            "available_duration_s": total_duration,
            "source_samples": stop - first,
            "output_samples": int(pcm.size),
            "highpass_hz": highpass_hz,
            "gain": applied_gain,
            "gain_mode": "auto" if gain is None else "fixed",
            "fixed_gain": gain,
        }
        print(
            f"Playing {actual_distance:g} m | PRF {source_rate} Hz | "
            f"start {info['start_s']:.3f} s | duration {info['duration_s']:.3f} s | "
            f"output {output_rate} Hz | high-pass {highpass_hz} | "
            f"gain {info['gain_mode']} ({applied_gain:g}x)"
        )
        self._play_pcm(pcm, output_rate)
        return info


def _main():
    parser = argparse.ArgumentParser(
        description="Play a DAS distance point using DASinterogator_offline (Windows)."
    )
    parser.add_argument("--distance", type=float, default=100.0, help="Distance in meters (100).")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (0).")
    parser.add_argument("--duration", type=float, default=None, help="Seconds to play (all remaining).")
    parser.add_argument("--data-dir", default=None, help="Data folder (rawData/data_180_240).")
    parser.add_argument("--volume", type=float, default=1.0, help="Auto peak level or fixed-mode volume multiplier, 0 to 1 (0.8).")
    parser.add_argument("--gain", type=float, default=None,
        help="Optional fixed gain before the volume multiplier; omit for automatic peak normalization.",
    )
    filtering = parser.add_mutually_exclusive_group()
    filtering.add_argument("--highpass", type=float, default=10.0, help="High-pass cutoff in Hz (10).")
    filtering.add_argument("--no-filter", action="store_true", help="Bypass the extra high-pass filter.")
    args = parser.parse_args()
    try:
        DASVoicePlayer(args.data_dir).play(
            distance_m=args.distance,
            start_s=args.start,
            duration_s=args.duration,
            highpass_hz=None if args.no_filter else args.highpass,
            volume=args.volume,
            gain=args.gain,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Playback failed: {exc}\n")


if __name__ == "__main__":
    _main()
