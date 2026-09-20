#./scripts/DASinterogator_offline.py
import sys
from pathlib import Path

# Project root = rawData/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os, glob
import numpy as np
import matplotlib.pyplot as plt
from scripts.parseDataHeader import parse_das_header as headerReader
# import struct
# import csv

# TODO: I must work on an other script that produces .dat files based on my needs
# TODO: make readPeriod and headerReader Rust python modules

class DASinterogator_offline:
    """
    Offline reader/processor for DAS interogator .dat files.

    File format
    ----------
    - Header: 256 bytes (128 x uint16, little-endian) describing acquisition parameters.
    - Payload: int16 samples arranged as (frames_per_file, num_traces).

    Coordinate system
    -----------------
    - Time axis: sampled at SCAN_RATE_HZ (a.k.a PRF / scan rate).
    - Distance axis: fiber distance in meters.
      Each trace corresponds to a spatial bin of size spatial_res_m.

    ROI (Region of Interest)
    ------------------------
    This class typically loads only a distance ROI [startDist, endDist] to reduce RAM and speed up
    subsequent processing (waterfall / FFT / event features).

    Typical workflow
    ----------------
    1) Read ROI for a period: data_ROI, t_axis, d_axis = readPeriod(period_s)
    2) Inspect waterfall: displayWaterFall(data_ROI, t_axis, d_axis)
    3) Compute FFT map: FFTdb, FFTfreqs = _FFTcalculator(data_TOI)
    4) Inspect freq-vs-distance: displayFreqvsDist(FFTdb, FFTfreqs, d_axis)
    """
    
    def __init__(
        self,
        baseDir: str,
        dataDir: str,
        startDist: float,
        endDist: float,
    ):
        # TODO: prepare new files of silent state and calculate mean based on them at initialization to use in .readperiod()
        PATTERN_CH1  = r"ch10__*.dat"
        PATTERN_CH2  = r"ch20__*.dat"
        HEADER_BYTES = 256
        self.DTYPE = np.int16
        
        self.MinDistance = startDist
        self.MaxDistance = endDist

        self.OUT_DIR = os.path.join(baseDir, "output")
        os.makedirs(self.OUT_DIR, exist_ok=True)
        
        self.paths = sorted( glob.glob(os.path.join(dataDir, PATTERN_CH1)))
        if not self.paths:
            raise FileNotFoundError("No DAT files found")

        parsedHeader = self._parse_headers(self.paths[0],HEADER_BYTES)
        
        self.FIBER_LENGTH_M = parsedHeader["Distance_m"]
        self.SCAN_RATE_HZ   = parsedHeader["prf_or_scan_rate_Hz"]
        self.FRAMES_OF_DATA = parsedHeader["frames_per_file"]

        self.spatial_points = parsedHeader["num_traces"]
        self.spatial_res_m  = parsedHeader["Dist_accuracy_m"]

        # ROI indices are stored as [min_idx, max_idx) (max exclusive).
        # Clamp after converting end distance to inclusive-bin then +1.
        self.min_idx = int(np.floor(self.MinDistance / self.spatial_res_m))
        self.min_idx = max(0, min(self.min_idx, self.spatial_points - 1))

        max_idx_inclusive = int(np.floor(self.MaxDistance / self.spatial_res_m))
        self.max_idx = max_idx_inclusive + 1
        self.max_idx = max(1, min(self.max_idx, self.spatial_points))

        if self.max_idx <= self.min_idx:
            self.max_idx = min(self.spatial_points, self.min_idx + 1)

        self.TimeDistSample_Data = None

    def _parse_headers(self,
                      filePath: str, 
                      headerSize: int = 256,
                      ):
        """
        Read and parse the DAS header from a single .dat file.

        Parameters
        ----------
        filePath : str
            Path to .dat file.
        headerSize : int
            Header size in bytes (expected 256).

        Returns
        -------
        dict
            Parsed header fields from scripts.parseDataHeader.parse_das_header().
            Returns {} on failure (you may prefer raising exceptions in production).
        """
        info = {}
        try:
            with open(filePath, "rb") as f:
                header = f.read(headerSize)
                info = headerReader(header)
                return info
        except:
            return info
    
    def readPeriod(self, period_s:int):
        # Question: What is the Effect of Normalization on the quality of events seperation like car vs pedestrain vs digging? 
        """
        Load a time period from disk (ROI-only) and apply basic preprocessing.

        The reader loads consecutive .dat files until it covers `period_s` seconds.
        It only keeps the distance ROI [MinDistance, MaxDistance] to reduce memory.

        Parameters
        ----------
        period_s : int
            Duration to load in seconds.

        Returns
        -------
        data_ROI : np.ndarray, float32, shape (N_time, N_dist)
            Normalized and de-meaned DAS samples for the distance ROI.
            Normalization uses int16 full-scale: sample/32768.
        t_axis : np.ndarray, float, shape (N_time,)
            Time values in seconds.
        d_axis : np.ndarray, float, shape (N_dist,)
            Distance values in meters for each spatial bin in the ROI.

        Notes
        -----
        Preprocessing performed:
        - scale int16 -> float in [-1, 1)
        - remove per-distance DC: subtract mean over time (fast, less robust than median)
        - remove global mean (center around 0)
        """
        DATA_LIMIT = int(1+(period_s*self.SCAN_RATE_HZ)/self.FRAMES_OF_DATA)

        roi_width = self.max_idx - self.min_idx ## Very Very Important for speed
        fullROI = np.empty((DATA_LIMIT*self.FRAMES_OF_DATA, roi_width), dtype=self.DTYPE)
        count = 0
        # TODO: (f.read() vs f.seek())? + struct **vs** np.fromfile()
        for path in self.paths:
            with open(path, "rb") as f:
                _ = f.read(256)
                data = np.fromfile(f, dtype=self.DTYPE).reshape(self.FRAMES_OF_DATA, self.spatial_points)
                fullROI[count:count+self.FRAMES_OF_DATA,:] = data[:, self.min_idx:self.max_idx]
            count += self.FRAMES_OF_DATA
            if count >= DATA_LIMIT*self.FRAMES_OF_DATA: break
        
        data_ROI = fullROI.astype(np.float32)
        data_ROI /= 32768.0
        t_axis = np.arange(data_ROI.shape[0]) / self.SCAN_RATE_HZ
        d_axis = np.arange(self.min_idx, self.max_idx, 1) * self.spatial_res_m
        # ---------------------------------------
        #         Pre-Process: Remove DC
        # ---------------------------------------
        # Remove DC per distance bin: mean over time.
        # (Median is more robust but significantly slower for large matrices.) 
        # mean is faster than median! but more sensitive to outliers, 
        # which is ok since we want to remove DC component and we do have enough samples for initialization
        # I Should do: Calculate mean for first 30s or 1m in initialization process ** useful for real time
        baseline  = np.mean(data_ROI, axis=0, keepdims=True)
        data_ROI -= baseline
        data_ROI  = data_ROI - np.mean(data_ROI)

        return data_ROI, t_axis, d_axis
    
    def displayWaterFall(self, 
                         data, 
                         t_axis, 
                         d_axis,
                         SaveMode = True
                         ):
        wf = 20*np.log10(np.abs(data)+1e-12)
        vmax = np.percentile(wf, 99.8)
        vmin = np.percentile(wf, 70)
        plt.figure(figsize=(12, 6))
        plt.title(f"Time vs Distance (Waterfall) — {int(self.MinDistance)} m TO {int(self.MaxDistance)} m")
        plt.xlabel("Distance (m)")
        plt.ylabel("Time (s)")
        im = plt.imshow(
            wf,
            aspect="auto",
            origin="lower",
            extent=((d_axis[0]), (d_axis[-1]), (t_axis[0]), (t_axis[-1])),
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(im, label="abs(signal) (normalized)")
        plt.tight_layout()
        if SaveMode:
            waterfall_path = os.path.join(self.OUT_DIR, f"waterfall_{int(self.MinDistance)}m_{int(self.MaxDistance)}m.png")
            plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()    
        
    def _FFTcalculator(self, 
                       TimeDistMatrix,
                       ):
        """
        Compute rFFT over time for each distance bin (column-wise FFT).

        Parameters
        ----------
        TimeDistMatrix : np.ndarray, shape (N_time, N_dist)
            Time-domain ROI segment. Rows are time samples, columns are distance bins.

        Returns
        -------
        mag_db : np.ndarray, shape (N_freq, N_dist)
            Magnitude spectrum in dB (20*log10(|X|) + DB_OFFSET).
            DB_OFFSET is a display/calibration offset; thresholds depend on this choice.
        freq_axis : np.ndarray, shape (N_freq,)
            Frequency axis in Hz for the rFFT bins.

        Implementation details
        ----------------------
        - Applies Hann window to reduce spectral leakage.
        - Uses rfft(axis=0): computes an FFT for each distance bin simultaneously.
        - Uses norm="forward" to reduce sensitivity to FFT length (amplitude scaling).
        """
        win = np.hanning(TimeDistMatrix.shape[0]).reshape(-1, 1)
        xw = TimeDistMatrix * win
        X = np.fft.rfft(xw, axis=0, norm="forward") # X is a Complex Value
        # keep norm="forward" to make features less sensitive to window length. 
        # Removing it will be more sensitive to window length.
        
        mag = np.abs(X) #* np.sqrt(2) # Remove * np.sqrt(2) for speedup
        EPS = 1e-12
        DB_OFFSET = 115
        mag_db = 20.0 * np.log10(mag + EPS) + DB_OFFSET
        freq_axis = np.fft.rfftfreq(TimeDistMatrix.shape[0], d=1.0 / self.SCAN_RATE_HZ)

        return mag_db, freq_axis, mag
    
    def displayFreqvsDist(self, 
                         mag_db,
                         freq_axis, 
                         d_axis,
                         freq_band_Hz = (2, 150),
                         SaveMode = True
                         ):
        
        minfreq_Hz, maxfreq_Hz = freq_band_Hz
        band_mask = (freq_axis >= minfreq_Hz) & (freq_axis <= maxfreq_Hz)
        if not np.any(band_mask):
            raise ValueError(
                f"Band mask is empty for band_hz={freq_band_Hz}. "
                f"Check PRF/time length; Nyquist is {self.SCAN_RATE_HZ/2:.2f} Hz."
            )
        FFTfreqs_band = freq_axis[band_mask]
        
        db_high = 20# np.percentile(mag_db[band_mask, :], 99.8)
        db_low  = 10# np.percentile(mag_db[band_mask, :], 95)
        plt.figure(figsize=(12, 6))
        plt.title(f"Frequency vs Distance — {int(self.MinDistance)} m TO {int(self.MaxDistance)} m (FFT over time)")
        plt.xlabel("Distance (m)")
        plt.ylabel("Frequency (Hz)")
        im2 = plt.imshow(
            mag_db[band_mask, :],
            aspect="auto",
            origin="lower",
            extent=(float(d_axis[0]), float(d_axis[-1]), float(FFTfreqs_band[0]), float(FFTfreqs_band[-1])),
            vmin=float(db_low),
            vmax=float(db_high),
        )
        plt.colorbar(im2, label="Magnitude (dB)")
        plt.tight_layout()
        if SaveMode:
            freqdist_path = os.path.join(self.OUT_DIR, f"freq_vs_distance—{int(self.MinDistance)}mTO{int(self.MaxDistance)}m.png")
            plt.savefig(freqdist_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        
        return mag_db

    def displaySinglePoint(self, 
                         mag_db,
                         freq_axis,
                         singlePoint_m: float,
                         minfreq_Hz = 5.0,
                         maxfreq_Hz = 200.0,
                         SaveMode = True
                         ):
        if (self.MaxDistance>=singlePoint_m>=self.MinDistance):
            singlePointIdx = int(singlePoint_m/self.spatial_res_m)-int(self.MinDistance/self.spatial_res_m)
        else:
            raise ValueError(f"singlePoint_m={singlePoint_m} m is out of the ROI range [{self.MinDistance}, {self.MaxDistance}] m.")
        
        minfreq = np.searchsorted(freq_axis, minfreq_Hz,side="left") # int(freq_axis.size * (minfreq_Hz / (self.SCAN_RATE_HZ/2)))
        maxfreq = np.searchsorted(freq_axis, maxfreq_Hz,side="right") # freq indexes
        freq_axis = freq_axis[minfreq:maxfreq]
        
        mag_db = mag_db[minfreq:maxfreq, singlePointIdx]

        df = freq_axis[1] - freq_axis[0]

        cs = np.empty(mag_db.size, dtype=mag_db.dtype)
        np.cumsum(mag_db, axis=0, out=cs)

        numAvg1 = int(round(5.0 / df)) # 5 + minfreq_Hz -> 7 Hz
        win_sum = cs[numAvg1:] - cs[:-numAvg1]
        MovingAverage01 = win_sum / numAvg1 
        print("Max-Min MA20 = ", np.max(MovingAverage01) - np.min(MovingAverage01))

        numAvg2 = int(round(10.0 / df)) # 10 + minfreq_Hz -> 12 Hz
        win_sum = cs[numAvg2:] - cs[:-numAvg2]
        MovingAverage02 = win_sum / numAvg2 
        print("Max-Min MA40 = ", np.max(MovingAverage02) - np.min(MovingAverage02))

        numAvg3 = int(round(20.0 / df)) # 20 + minfreq_Hz -> 22 Hz
        win_sum = cs[numAvg3:] - cs[:-numAvg3]
        MovingAverage03 = win_sum / numAvg3 
        print("Max-Min MA100 = ", np.max(MovingAverage03) - np.min(MovingAverage03))

        numAvg4 = int(round(45.0 / df)) # 45 + minfreq_Hz -> 47 Hz
        win_sum = cs[numAvg4:] - cs[:-numAvg4]
        MovingAverage04 = win_sum / numAvg4 
        print("Max-Min MA200 = ", np.max(MovingAverage04) - np.min(MovingAverage04))

        print("Max(MA100) - Mean(Spectrum) = ", np.max(MovingAverage03) - np.mean(mag_db))
        print("Total Mean: ", np.mean(mag_db))

        plt.figure()
        plt.title(f"Scaled Power - frequency Domain @{singlePoint_m:.2f}m")

        plt.plot(freq_axis, mag_db)
        plt.plot(freq_axis[numAvg1:], MovingAverage01) # freq_axis[numAvg1-int(numAvg1/2):-int(numAvg1/2)]
        plt.plot(freq_axis[numAvg2:], MovingAverage02) # freq_axis[numAvg2-int(numAvg2/2):-int(numAvg2/2)]
        plt.plot(freq_axis[numAvg3:], MovingAverage03) # freq_axis[numAvg3-int(numAvg3/2):-int(numAvg3/2)]
        plt.plot(freq_axis[numAvg4:], MovingAverage04) # freq_axis[numAvg4-int(numAvg4/2):-int(numAvg4/2)]
        plt.plot(freq_axis, np.zeros(freq_axis.size) + np.mean(mag_db))
        plt.plot(freq_axis, np.zeros(freq_axis.size)+np.max(mag_db))
        # plt.plot(freq_axis, np.zeros(freq_axis.size)+15)

        plt.grid(True)
        plt.ylim(0,35)
        if SaveMode:
            singlePointFreqPath = os.path.join(self.OUT_DIR, f"SinglePoint{singlePoint_m}m.png")
            plt.savefig(singlePointFreqPath, dpi=300, bbox_inches="tight")

        plt.show()
        plt.close()
        return "OK"
    
    def displayPeakAnimation_PlusPeakDetection(self,
                             data,
                             d_axis,
                             FFTtimeStep_s = 1,
                             FFTtimeWindow_s = 2,
                             HzAverage = 10,
                             freq_band_Hz = (2, 150),
                             z_threshold = 10,
                             min_cluster_bins = 6
                             ):

        maxTime = data.shape[0] // self.SCAN_RATE_HZ

        plt.ion()
        fig, ax = plt.subplots()
        line01, = ax.plot([],[], 'r-')
        line02, = ax.plot([],[], 'b-')

        ax.set_xlim(d_axis.min(), d_axis.max())
        ax.set_ylim(-1, 25) # 110-85
        ax.grid(True)
        
        minfreq_Hz, maxfreq_Hz = freq_band_Hz

        # FFTdb: (N_f, N_pos)  dB values
        df = 1/FFTtimeWindow_s                 # Hz per bin
        W  = int(round(HzAverage / df))        # window width in bins (≈ 10 Hz)
        W  = max(W, 1)
        mu = 1e-12
        sigma = 1e-12

        for t in np.arange(0,maxTime-FFTtimeWindow_s,FFTtimeStep_s):
            startSecond = t
            data_TOI = data[int(startSecond*self.SCAN_RATE_HZ):int((startSecond+FFTtimeWindow_s)*self.SCAN_RATE_HZ), :]
            # data_TOI: (N_timeSample, N_pos)
            FFTdb, FFTfreqs, mag = self._FFTcalculator(TimeDistMatrix=data_TOI)

            ###
            minfreqIdx = np.searchsorted(FFTfreqs, minfreq_Hz, side="left")
            maxfreqIdx = np.searchsorted(FFTfreqs, maxfreq_Hz, side="right")

            # FFTfreqs = FFTfreqs[minfreqIdx:maxfreqIdx] # e.g. 2.0 2.5 3.0 ... 150.0 Hz
            FFTdb = FFTdb[minfreqIdx:maxfreqIdx, :]
            mag = mag[minfreqIdx:maxfreqIdx, :]
            ###

            cs = np.empty((FFTdb.shape[0] , FFTdb.shape[1]), dtype=FFTdb.dtype)
            np.cumsum(FFTdb, axis=0, out=cs)
            win_sum = cs[W:, :] - cs[:-W, :]
            # Sum each consecutive W-frequency-bin block for every distance position.
            # Using cumsum keeps this O(N_f * N_pos) instead of recomputing each window.
            # Next, divide by W to get the sliding average and take max over frequency.
            # for more info look at ./test/testcumsum.py
            win_avg = win_sum / W       # (N_f - W, N_pos)
            
            cs2 = np.empty((mag.shape[0] , mag.shape[1]), dtype=mag.dtype)
            np.cumsum(mag, axis=0, out=cs2)
            win_sum2 = cs2[W:, :] - cs2[:-W, :]
            win_avg2 = win_sum2 / W       # (N_f - W, N_pos)

            SlidingAverage_MAX = np.max(win_avg, axis=0)   # (N_pos,)
            SlidingAverage_MAX_mag = np.max(win_avg2, axis=0)   # (N_pos,)
            # if t<10:
            mu = (np.mean(SlidingAverage_MAX_mag) + 1*mu) / 2
            sigma = (np.std(SlidingAverage_MAX_mag) + 1e-12 + 1*sigma) / 2
            
            z = 5*(SlidingAverage_MAX_mag - mu) / (sigma + 1e-12)

            line01.set_data(d_axis, SlidingAverage_MAX)
            line02.set_data(d_axis, z)
            
            # TODO: clustering and reporting events based on clusters 
            if np.max(SlidingAverage_MAX) > z_threshold:
                print(f"\nThere is an Event at distance {d_axis[np.argmax(SlidingAverage_MAX)]:.2f} m at time {t} s with max Sliding Average of {np.max(SlidingAverage_MAX):.2f} dB")
            
            active = np.where(z > z_threshold/2)[0]
            events = []
            clusters = np.split(active, np.where(np.diff(active) != 1)[0] + 1) # np.diff: out[i] = a[i+1] - a[i]
            for cluster in clusters:
                if cluster.size < min_cluster_bins:
                    continue
                d0 = int(np.mean(cluster))
                events.append({
                    "distance_index": d0,
                    "distance_m": np.round(d_axis[d0],2),
                    "z_score": np.round(float(np.max(z[cluster])),4),
                    # "band_hz_low": float(minfreq_Hz),
                    # "band_hz_high": float(maxfreq_Hz),
                })
                print(events[-1])

            plt.draw()
            plt.pause(0.1)

        plt.ioff()
        plt.close()

    def detect_events_timeDomain(self,
                                 data,
                                 d_axis,
                                 t_axis,
                                 sigma_threshold=10.0,
                                 min_cluster_size=6,
                                 speedFactor = 2
                                 ):
        """
        Returns list of events:
        [
          {
            time_index,
            distance_index,
            time_sec,
            distance_m,
            amplitude
          },
          ...
        ]
        """
        plt.ion()
        fig, ax = plt.subplots()
        line01, = ax.plot([],[], 'r-')
        line02, = ax.plot([],[], 'b-')

        ax.set_xlim(d_axis.min(), d_axis.max())
        ax.set_ylim(0, 0.3)
        ax.grid(True)
        plt.title("Time-Domain Peak Detection (Red=Envelope, Blue=Threshold)")
        plt.xlabel("Distance (m)")
        plt.ylabel("Normalized Amplitude")
        plt.tight_layout()
        ax.legend(["Envelope", f"Threshold (mu + {sigma_threshold}*sigma)"])

        env = np.abs(data)
        env = env / (np.max(env) + 1e-12)
        env *= env
        
        mu = np.mean(env, axis=0)
        sigma = np.std(env, axis=0) + 1e-12
        threshold = mu + sigma_threshold * sigma
        line02.set_data(d_axis, threshold)

        event_mask = env > threshold[:]
        
        events = []

        T, D = event_mask.shape

        for t in range(0,T,speedFactor):
            active = np.where(event_mask[t])[0]
            if active.size == 0:
                continue

            # cluster contiguous distance bins
            clusters = np.split(active, np.where(np.diff(active) != 1)[0] + 1)

            for c in clusters:
                if len(c) < min_cluster_size:
                    continue

                d0 = int(np.mean(c))
                amp = float(np.max(env[t, c]))

                events.append({
                    "time_index": np.round(t_axis[t],1),
                    "distance_m": np.round(d_axis[d0],2),
                    "amplitude": amp
                })

            line01.set_data(d_axis, env[t,:])
            
            plt.draw()
            plt.pause(0.0004)

        plt.ioff()
        plt.close()

        # for e in events:
        #     print(e)
        # return events

    def detect_band_energy_Plus_freqDistAnimation(
            self,
            data,
            d_axis,
            FFTtimeStep_s = 1,
            FFTtimeWindow_s = 2,
            freq_band_Hz = (2, 150),
            z_threshold=2.0,
            min_cluster_bins=5,
            fixed_scale = False
        ):
        # TODO: research on events effect on frequency bands and not just specific freqs
        """
        DAS-style event detection using band-limited spectral energy.

        Input:
        data_time_distance: shape (T, D)
            T: stacked slow-time shots
            D: spatial bins (distance)

        Steps:
        1) Hann window in time to reduce spectral leakage
        2) rFFT along time axis for each spatial bin
        3) integrate |FFT| over frequency band
        4) z-score across distance bins and threshold
        5) cluster contiguous bins into events

        Output:
        events: list[dict]
            {
            distance_index, distance_m, z_score, band_energy, band_hz_low, band_hz_high
            }
        """
        maxTime = data.shape[0] // self.SCAN_RATE_HZ
        minfreq_Hz, maxfreq_Hz = freq_band_Hz

        ###############################################
        ###                Plot Set Up              ###
        ###############################################
        startSecond = 2.5
        timeWindowDuration_s = 2

        data_TOI = data[int(startSecond*self.SCAN_RATE_HZ):int((startSecond+timeWindowDuration_s)*self.SCAN_RATE_HZ), :]
        FFTdb, FFTfreqs, _ = self._FFTcalculator(data_TOI)
        band_mask = (FFTfreqs >= minfreq_Hz) & (FFTfreqs <= maxfreq_Hz)
        if not np.any(band_mask):
            raise ValueError(
                f"Band mask is empty for band_hz={freq_band_Hz}. "
                f"Check PRF/time length; Nyquist is {self.SCAN_RATE_HZ/2:.2f} Hz."
            )
        FFTfreqs_band = FFTfreqs[band_mask]
        print(band_mask.sum(), "frequency bins in the band of interest:", FFTfreqs_band[0], "to", FFTfreqs_band[-1], "Hz\n")
        
        plt.ion()
        fig, ax = plt.subplots()
        db_high = np.percentile(FFTdb[band_mask, :], 99.8)
        db_low = np.percentile(FFTdb[band_mask, :], 95)
        heatmap = ax.imshow(
                FFTdb[band_mask, :],
                aspect="auto",
                origin="lower",
                extent=(float(d_axis[0]), float(d_axis[-1]), float(FFTfreqs_band[0]), float(FFTfreqs_band[-1])),
                vmin=float(db_low),
                vmax=float(db_high),
            )
        plt.colorbar(heatmap, label="Magnitude (dB)")
        plt.title(f"Frequency vs Distance — {int(self.MinDistance)} m TO {int(self.MaxDistance)} m (FFT over time)")
        plt.xlabel("Distance (m)")
        plt.ylabel("Frequency (Hz)")
        plt.grid(True)
        ###############################################
        ###
        ###############################################
        mu = 0
        sigma = 0

        for t in np.arange(0,maxTime-FFTtimeWindow_s,FFTtimeStep_s):
            startSecond = t
            data_TOI = data[int(startSecond*self.SCAN_RATE_HZ):int((startSecond+FFTtimeWindow_s)*self.SCAN_RATE_HZ), :]
            # data_TOI: (N_timeSample, N_pos)
            FFTdb, FFTfreqs,_ = self._FFTcalculator(TimeDistMatrix=data_TOI)
            FFTdbFOI = FFTdb[band_mask, :]
            heatmap.set_data(FFTdbFOI)
            if not fixed_scale:
                db_high = np.percentile(FFTdbFOI, 99.8)
                db_low = np.percentile(FFTdbFOI, 95)
                heatmap.set_clim(db_low, db_high)
                
            band_energy = FFTdbFOI.sum(axis=0) # (N_distBin,)
            # print(FFTdb[band_mask, :].shape) # (297, 189) ( (maxfreq_Hz-minfreq_Hz)*2+1, ROI dist Bins )
            mu = (np.mean(band_energy) + mu)/2 # value # Question: Is't it better to calculate at initialization? or here in loop is ok?
            sigma = (np.std(band_energy) + 1e-12 + sigma)/2 # value
            # print("band_energy.shape: ", band_energy.shape)
            # print("mu:    ", mu)
            # print("sigma: ", sigma)
            z = (band_energy - mu) / sigma # (N_distBin,)
            # band_energy - mu:  is a kind of comparision with neighbor distance bins
            # TODO: write phisical interpretation of z in comments 

            active = np.where(z >= z_threshold)[0] # -> Indexes List
            # print("active:\n", active, active.shape)
            events = []
            # if active.size == 0:
            #     return events
            # Cluster contiguous distance bins
            clusters = np.split(active, np.where(np.diff(active) != 1)[0] + 1) # np.diff: out[i] = a[i+1] - a[i]

            for cluster in clusters:
                if cluster.size < min_cluster_bins:
                    continue
                d0 = int(np.mean(cluster))
                events.append({
                    "distance_index": d0,
                    "distance_m": np.round(d_axis[d0],2),
                    "z_score": np.round(float(np.max(z[cluster])),4),
                    "band_energy": float(np.max(band_energy[cluster])/(maxfreq_Hz-minfreq_Hz)),
                    # "band_hz_low": float(minfreq_Hz),
                    # "band_hz_high": float(maxfreq_Hz),
                })
            if (len(events) != 0) and (len(clusters)>=1):
                for e in events:
                    print(t,'s\t', e)
                print('\n')

            plt.draw()
            plt.pause(0.1)
        plt.ioff()
        plt.close()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Only for this project in which current file located in ./scripts:
    BASE_DIR = os.path.dirname(BASE_DIR) 

    DATA_DIR = os.path.join(BASE_DIR, "data_180_240")

    MIN_DIST_M = 175.0
    MAX_DIST_M = 250.0

    myDAS = DASinterogator_offline(baseDir = BASE_DIR, 
                                   dataDir = DATA_DIR,
                                   startDist = MIN_DIST_M,
                                   endDist = MAX_DIST_M
                                   )
    
    maxTime = 130
    data_ROI, t_axis, d_axis = myDAS.readPeriod(period_s=maxTime) 
    
    x = 1 # For simulating lower PRF
    if x> 1:
        myDAS.SCAN_RATE_HZ /= x
        data_ROI = data_ROI[::x,:]
        t_axis = t_axis[::x]
    ## Max period_s to kill 110 -> now it's ok upto 137s, at the bigging it was less than 50s
    
    ### data_ROI: signed floating point value

    myDAS.displayWaterFall(data_ROI, t_axis, d_axis, SaveMode=False)

    # startSecond = 6
    # timeWindowDuration_s = 2
    # data_TOI = data_ROI[int(startSecond*myDAS.SCAN_RATE_HZ):int((startSecond+timeWindowDuration_s)*myDAS.SCAN_RATE_HZ), :]
    # FFTdb, FFTfreqs,_ = myDAS._FFTcalculator(data_TOI)
    # myDAS.displayFreqvsDist(FFTdb, FFTfreqs, d_axis,freq_band_Hz=(0,750), SaveMode=True)
    # myDAS.displaySinglePoint(FFTdb, FFTfreqs, 235, minfreq_Hz=3, maxfreq_Hz=175.5, SaveMode=False)

    # myDAS.displayPeakAnimation_PlusPeakDetection(data_ROI, d_axis,freq_band_Hz=(5,145.5))
    # myDAS.detect_band_energy_Plus_freqDistAnimation(data_ROI, d_axis, freq_band_Hz=(3,200), fixed_scale=True)
    # myDAS.detect_events_timeDomain(data_ROI,d_axis,t_axis)
    pass
