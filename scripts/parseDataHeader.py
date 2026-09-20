#./scripts/parseDataHeader.py
import struct
from datetime import datetime

# try:
#     from ._parse_data_header_rs import parse_das_header as _parse_das_header_rs  # type: ignore
# except Exception:
#     try:
#         from _parse_data_header_rs import parse_das_header as _parse_das_header_rs  # type: ignore
#     except Exception:
#         _parse_das_header_rs = None

def parse_das_header(header_bytes: bytes) -> dict:
    """
    Parse a 256-byte DAS header and return a dictionary of interpreted fields.

    # Uses a Rust extension module (`_parse_data_header_rs`) when available for
    # faster and safer binary decoding. Falls back to pure Python otherwise.

    Assumptions (validated from data):
      - Header length = 256 bytes (128*uint16)
      - Layout = 128 uint16 values, little-endian

    Parameters
    ----------
    header_bytes : bytes
        Exactly 256 bytes from the beginning of a DAS .dat file.

    Returns
    -------
    dict
        Parsed header information with validated fields + raw access.
        
        magic_0, magic_1
        
        timestamp, timestamp_fields
        
        channel_ID
        
        sample_rate_MHz
        Dist_accuracy_m
        Distance_m
        num_traces, last_trace_index
        
        pulse_width
        prf_or_scan_rate_Hz
        frames_per_file
        
        raw_u16
    """

    if len(header_bytes) != 256:
        raise ValueError(f"Header must be exactly 256 bytes, got {len(header_bytes)}")

    # unpack to 128 uint16 values (little-endian)
    u16 = list(struct.unpack("<128H", header_bytes))
    
    header = {}

    # ----------------------
    # --- Identification ---
    # ----------------------
    header["magic_0"] = u16[0] # 2204 || 0x089C || 0b 0000 1000 1001 1100
    header["magic_1"] = u16[1] #  256 || 0x0100 || 0b 0000 0001 0000 0000
    
    if header["magic_0"] != 2204 or header["magic_1"] != 256:
        raise ValueError(f"Unexpected header magic: {header['magic_0']}, {header['magic_1']}")

    # -------------------------------
    # --- Timestamp (from header) ---
    # -------------------------------
    # YY MM DD HH MM SS mmm
    try:
        year = 2000 + u16[2]
        month = u16[3]
        day = u16[4]
        hour = u16[5]
        minute = u16[6]
        second = u16[7]
        millisecond = u16[8]

        header["timestamp"] = datetime(
            year, month, day,
            hour, minute, second,
            millisecond * 1000
        )
    except Exception:
        # keep partial info if timestamp is malformed
        header["timestamp"] = None

    header["timestamp_fields"] = {
        "year_yy": u16[2],
        "month": u16[3],
        "day": u16[4],
        "hour": u16[5],
        "minute": u16[6],
        "second": u16[7],
        "millisecond": u16[8],
    }

    # ------------------------------
    # --- Acquisition parameters ---
    # ------------------------------
    # u16[9] = 10
    # u16[10] = 1
    header["channel_ID"] = u16[11] if u16[11] != 0 else None
    # u16[12:17] = 0
    
    header["sample_rate_MHz"] = None if (u16[17]==0 and u16[18]==0) else (u16[17] + (u16[18]/10000.0))
  
    header["Dist_accuracy_m"] = (100 / header["sample_rate_MHz"]) if header["sample_rate_MHz"] else None
    # u16[19:24] = [0, 0, 0, 65535, 15]
    header["pulse_width"] = u16[24] if u16[24] != 0 else None
    # u16[25:27] = [65535, 0]
    # u16[30:37] = [0, 0, 0, 0, 0, 0, 0]
    header["prf_or_scan_rate_Hz"] = u16[37] if u16[37] != 0 else None
    # u16[38] = 0
    header["frames_per_file"] = u16[39] if u16[39] != 0 else None
    
    # --------------------------
    # --- Spatial parameters ---
    # --------------------------
    header["last_trace_index"] = u16[27] if u16[27] != 0 else None
    # u16[28] = 0
    header["num_traces"] = u16[29] if u16[29] != 0 else None # = Distance_m * (sampleRate_MHz/100)
    header["Distance_m"] = ( 100 * (header["num_traces"] / header["sample_rate_MHz"]) ) if header["num_traces"] else None

    # ------------------
    # --- Raw access ---
    # ------------------
    header["raw_u16"] = u16

    # u16[40:127] = 0
    return header


if __name__ == "__main__":
    import glob
    import os
    
    # Only for this project in which current file located in ./scripts:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(BASE_DIR)
    DATA_DIR = os.path.join(BASE_DIR, "dataHeader")
    
    PATTERN_CH1 = r"ch10__*.dat"
    PATTERN_CH2 = r"ch20__*.dat"
    
    paths10 = sorted( glob.glob(os.path.join(DATA_DIR, PATTERN_CH1)) )
    paths20 = sorted( glob.glob(os.path.join(DATA_DIR, PATTERN_CH2)) )
    paths = paths10 + paths20
    print(f"Found {len(paths)} files matching patterns in {DATA_DIR}")
    
    if len(paths) == 0:
        raise FileNotFoundError("No files found. Please check the dataHeader directory and filename patterns.")
    
    for path in paths:
        with open(path, "rb") as f:
            header_bytes = f.read(256)

        hdr = parse_das_header(header_bytes)

        print(hdr["raw_u16"][:41])

        
# Header layout notes (empirically derived from DAS interogator files):
#   [0:2]   magic words
#   [2:9]   timestamp fields (YY MM DD HH MM SS mmm)
#   [11]    channel id
#   [17:19] sample rate MHz as int + fractional(1/10000)
#   [24]    pulse width (device-specific units)
#   [37]    PRF / scan rate (Hz)
#   [39]    frames per file
#   [27]    last trace index (num_traces-1)
#   [29]    num_traces (spatial points)
# Many other indices appear reserved/constant in the provided captures.
        
# [2204, 256, 26, 1, 11, 10, 26, 57, 795, 10, 1, 1, 0, 0, 0, 0, 0, 250,    0, 0, 0, 0, 65535, 15,  50, 65535, 0, 4999, 0,  5000, 0, 0, 0, 0, 0, 0, 0, 1000, 0, 1024, 0]
# [2204, 256, 26, 1, 11, 10, 29,  8, 640, 10, 1, 1, 0, 0, 0, 0, 0, 250,    0, 0, 0, 0, 65535, 15,  50, 65535, 0, 4999, 0,  5000, 0, 0, 0, 0, 0, 0, 0,  500, 0, 1024, 0]
# [2204, 256, 26, 1, 11, 10, 30, 48, 650, 10, 1, 1, 0, 0, 0, 0, 0, 250,    0, 0, 0, 0, 65535, 15,  84, 65535, 0, 4999, 0,  5000, 0, 0, 0, 0, 0, 0, 0, 1024, 0, 1024, 0]
# [2204, 256, 26, 1, 11, 10, 33, 24, 896, 10, 1, 1, 0, 0, 0, 0, 0,  31, 2500, 0, 0, 0, 65535, 15, 100, 65535, 0, 6249, 0,  6250, 0, 0, 0, 0, 0, 0, 0, 1000, 0, 1024, 0]
# [2204, 256, 26, 1, 11, 10, 35,  1, 623, 10, 1, 1, 0, 0, 0, 0, 0,  50,    0, 0, 0, 0, 65535, 15, 100, 65535, 0, 9999, 0, 10000, 0, 0, 0, 0, 0, 0, 0, 1000, 0, 1024, 0]
# [2204, 256, 26, 1, 11, 15,  0,  2, 795, 10, 1, 1, 0, 0, 0, 0, 0,  25,    0, 0, 0, 0, 65535, 15, 500, 65535, 0, 9999, 0, 10000, 0, 0, 0, 0, 0, 0, 0,  800, 0,  512, 0]
# [2204, 256, 26, 1, 11, 10, 38, 40, 824, 10, 1, 2, 0, 0, 0, 0, 0,  25,    0, 0, 0, 0, 65535, 15, 110, 65535, 0, 4999, 0,  5000, 0, 0, 0, 0, 0, 0, 0, 1500, 0, 1024, 0]
