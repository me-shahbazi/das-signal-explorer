# fiberhome_realtime_headless.py
"""
Headless Real-time FiberHome DAS processor (NO GUI, NO Tkinter).

What it does:
- Watches WATCH_DIR for new .dat files
- For each file:
  - Parses payload as (FRAMES_OF_DATA, spatial_points)
  - DC/background subtract per spatial channel
  - Generates and saves:
      1) Waterfall (time vs distance) PNG
      2) Frequency vs Distance PNG (FFT over time axis)
      3) Dominant frequency per location CSV
      4) Simple event list CSV

This version uses matplotlib Agg backend, so it works even if Tcl/Tk (tkinter) is missing.
"""

import os
import time
import numpy as np
import csv

# --- Force headless backend BEFORE importing pyplot ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ============================================================
# PARAMETERS (FROM YOUR JSON)
# ============================================================
SAMPLE_RATE_MHZ = 250      # informational only
FIBER_LENGTH_M = 2000.0
PULSE_WIDTH = 50           # informational only
SCAN_RATE_HZ = 1024.0      # PRF / slow-time sampling
FRAMES_OF_DATA = 1024      # time shots per file

HEADER_BYTES = 256
DTYPE = np.int16           # keep int16 (your stats looked consistent)

EPS = 1e-12

# ============================================================
# DIRECTORIES
# ============================================================
BASE_DIR = os.path.dirname(__file__)
WATCH_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# IO + Processing
# ============================================================
def read_dat_file(path: str):
    """
    Read FiberHome .dat file and return:
      data: (FRAMES_OF_DATA, spatial_points) float32
      spatial_points: inferred from payload size
      spatial_res_m: FIBER_LENGTH_M / spatial_points
    """
    with open(path, "rb") as f:
        f.seek(HEADER_BYTES)
        raw = np.fromfile(f, dtype=DTYPE)

    if raw.size % FRAMES_OF_DATA != 0:
        raise ValueError(
            f"Payload samples ({raw.size}) not divisible by framesofDATA ({FRAMES_OF_DATA})."
        )

    spatial_points = raw.size // FRAMES_OF_DATA
    spatial_res_m = FIBER_LENGTH_M / spatial_points

    data = raw.reshape(FRAMES_OF_DATA, spatial_points).astype(np.float32)

    # Remove static backscatter level (DC) per distance bin
    baseline = np.mean(data, axis=0, keepdims=True)
    data = data - baseline

    return data, spatial_points, spatial_res_m


def compute_fft_map(data: np.ndarray):
    """
    FFT over time axis (axis=0) for each spatial bin.
    Returns:
      freqs (Hz): (n_freq_bins,)
      mag: (n_freq_bins, spatial_points)
    """
    win = np.hanning(data.shape[0]).reshape(-1, 1)
    data_win = data * win

    fft = np.fft.rfft(data_win, axis=0)
    mag = np.abs(fft)

    freqs = np.fft.rfftfreq(data.shape[0], d=1.0 / SCAN_RATE_HZ)
    return freqs, mag


def export_dominant_freq(freqs, mag, spatial_res_m, filename_base):
    """
    For each spatial point, find dominant frequency (max magnitude).
    CSV columns: distance_m, dominant_freq_hz
    """
    dom_idx = np.argmax(mag, axis=0)
    dom_freq = freqs[dom_idx]

    csv_path = os.path.join(OUTPUT_DIR, filename_base + "_dominant_freq.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["distance_m", "dominant_freq_hz"])
        for i, f0 in enumerate(dom_freq):
            w.writerow([i * spatial_res_m, float(f0)])

    print("Saved:", csv_path)


def detect_events_envelope(data, sigma=6.0, min_cluster_size=5):
    """
    Simple time-local event detector (per file):
    - envelope = abs(data)
    - threshold per distance bin: mean + sigma*std
    - for each time shot, find contiguous distance clusters above threshold

    Returns list of tuples:
      (time_sec, distance_m, amplitude)
    """
    env = np.abs(data)
    mu = np.mean(env, axis=0)
    std = np.std(env, axis=0) + 1e-6
    th = mu + sigma * std

    mask = env > th[None, :]

    events = []
    T, D = mask.shape
    for t in range(T):
        active = np.where(mask[t])[0]
        if active.size == 0:
            continue

        clusters = np.split(active, np.where(np.diff(active) != 1)[0] + 1)
        for c in clusters:
            if c.size < min_cluster_size:
                continue
            d0 = int(np.mean(c))
            amp = float(np.max(env[t, c]))
            events.append((t / SCAN_RATE_HZ, d0, amp))  # distance index returned; convert later

    return events


def save_waterfall_png(data, spatial_points, spatial_res_m, filename_base):
    """
    Save Waterfall image: abs(signal) vs (time, distance)
    """
    t_axis = np.arange(data.shape[0]) / SCAN_RATE_HZ
    d_axis = np.arange(spatial_points) * spatial_res_m

    plt.figure(figsize=(12, 6))
    plt.title("Waterfall (Time vs Distance) | abs(signal)")
    plt.xlabel("Distance (m)")
    plt.ylabel("Time (s)")
    plt.imshow(
        np.abs(data),
        aspect="auto",
        origin="lower",
        extent=[d_axis[0], d_axis[-1], t_axis[0], t_axis[-1]],
    )
    plt.colorbar(label="abs(signal) (arb. units)")
    plt.tight_layout()

    out_img = os.path.join(OUTPUT_DIR, filename_base + "_waterfall.png")
    plt.savefig(out_img, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out_img)


def save_freq_vs_distance_png(freqs, mag, spatial_points, spatial_res_m, filename_base):
    """
    Save Frequency vs Distance image (in dB):
      x: distance (m)
      y: frequency (Hz)
      color: 20log10(|FFT|)
    """
    d_axis = np.arange(spatial_points) * spatial_res_m
    mag_db = 20.0 * np.log10(mag + EPS)

    plt.figure(figsize=(12, 6))
    plt.title("Frequency vs Distance (FFT over time axis)")
    plt.xlabel("Distance (m)")
    plt.ylabel("Frequency (Hz)")
    plt.imshow(
        mag_db,
        aspect="auto",
        origin="lower",
        extent=[d_axis[0], d_axis[-1], freqs[0], freqs[-1]],
    )
    plt.colorbar(label="Magnitude (dB)")
    plt.tight_layout()

    out_img = os.path.join(OUTPUT_DIR, filename_base + "_freq_vs_distance.png")
    plt.savefig(out_img, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out_img)


def export_events_csv(events, spatial_res_m, filename_base):
    """
    Save detected events:
      time_sec, distance_m, amplitude
    """
    out_csv = os.path.join(OUTPUT_DIR, filename_base + "_events.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "distance_m", "amplitude"])
        for time_sec, d_idx, amp in events:
            w.writerow([time_sec, d_idx * spatial_res_m, amp])
    print("Saved:", out_csv)


def process_one_file(path: str):
    """
    Full per-file pipeline:
      parse -> waterfall -> FFT map -> freq-distance -> dominant freq -> events
    """
    base = os.path.splitext(os.path.basename(path))[0]

    data, spatial_points, spatial_res_m = read_dat_file(path)

    # 1) Waterfall
    save_waterfall_png(data, spatial_points, spatial_res_m, base)

    # 2) FFT map and freq-distance plot
    freqs, mag = compute_fft_map(data)
    save_freq_vs_distance_png(freqs, mag, spatial_points, spatial_res_m, base)

    # 3) Dominant frequency per location
    export_dominant_freq(freqs, mag, spatial_res_m, base)

    # 4) Simple events (per-file)
    events = detect_events_envelope(data, sigma=6.0, min_cluster_size=5)
    if events:
        export_events_csv(events, spatial_res_m, base)
    else:
        # still create an empty CSV for consistency
        export_events_csv([], spatial_res_m, base)


# ============================================================
# DIRECTORY WATCHER
# ============================================================
class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if not event.src_path.lower().endswith(".dat"):
            return

        # Wait to ensure file is fully written
        time.sleep(0.5)

        try:
            print("New file:", event.src_path)
            process_one_file(event.src_path)
        except Exception as e:
            print("Error processing file:", event.src_path, "|", e)


def main():
    print("Headless watcher started")
    print("Watching folder:", os.path.abspath(WATCH_DIR))
    print(f"Configured scanRate_Hz={SCAN_RATE_HZ}, framesofDATA={FRAMES_OF_DATA}, fiber_length={FIBER_LENGTH_M} m")
    print("Outputs go to:", os.path.abspath(OUTPUT_DIR))

    # Process existing files once at startup
    for fname in sorted(os.listdir(WATCH_DIR)):
        if fname.lower().endswith(".dat"):
            fpath = os.path.join(WATCH_DIR, fname)
            try:
                print("Processing existing:", fpath)
                process_one_file(fpath)
            except Exception as e:
                print("Error processing existing file:", fpath, "|", e)

    # Watch for new files
    observer = Observer()
    observer.schedule(Handler(), WATCH_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
