"""
RF Vision — CSI Dataset Collector

Usage:

    python collect.py idle
    python collect.py motion

Files are saved as:

    data/idle_01.csv
    data/idle_02.csv
    data/motion_01.csv
    data/motion_02.csv

Each file represents ONE recording session.

Do not mix sessions during recording.
"""

import csv
import os
import sys
import time
import struct

import numpy as np
import serial


# ============================================================
# CONFIGURATION
# ============================================================

PORT = "COM3"
BAUD = 115200

RECORD_DURATION_SEC = 300

OUTPUT_DIR = "data"

MAGIC = b"\xA5\x5A"

MAX_CSI_LEN = 256

HEADER_AFTER_MAGIC = 22

# ------------------------------------------------------------
# Change this if you later verify a different acquisition rate.
# Current measured rate: ~7.83 Hz.
# ------------------------------------------------------------

PROCESSING_RATE_HZ = 8


# ============================================================
# BINARY PARSER
# ============================================================

def read_exact(ser, n):
    """
    Read exactly n bytes from serial.
    """

    data = bytearray()

    while len(data) < n:

        chunk = ser.read(
            n - len(data)
        )

        if not chunk:
            return None

        data.extend(chunk)

    return bytes(data)


def crc16_ccitt(data):
    crc = 0xFFFF

    for byte in data:

        crc ^= byte << 8

        for _ in range(8):

            if crc & 0x8000:
                crc = (
                    (crc << 1) ^
                    0x1021
                ) & 0xFFFF

            else:
                crc = (
                    crc << 1
                ) & 0xFFFF

    return crc


def read_csi_packet(ser):
    """
    Read one binary CSI packet.

    Returns:
        dict
        or None
    """

    # Search for magic.
    while True:

        first = ser.read(1)

        if not first:
            return None

        if first != b"\xA5":
            continue

        second = ser.read(1)

        if second == b"\x5A":
            break

    header = read_exact(
        ser,
        HEADER_AFTER_MAGIC
    )

    if header is None:
        return None

    (
        version,
        sequence,
        timestamp_us,
        rssi,
        channel,
        csi_len,
        flags,
        queue_drops,
    ) = struct.unpack(
        "<BIqbbHBI",
        header
    )

    if version != 1:
        return None

    if (
        csi_len <= 0 or
        csi_len > MAX_CSI_LEN
    ):
        return None

    csi = read_exact(
        ser,
        csi_len
    )

    if csi is None:
        return None

    crc_bytes = read_exact(
        ser,
        2
    )

    if crc_bytes is None:
        return None

    received_crc = struct.unpack(
        "<H",
        crc_bytes
    )[0]

    packet_without_crc = (
        MAGIC +
        header +
        csi
    )

    calculated_crc = crc16_ccitt(
        packet_without_crc
    )

    if calculated_crc != received_crc:
        return None

    return {
        "sequence": sequence,
        "timestamp_us": timestamp_us,
        "rssi": rssi,
        "channel": channel,
        "csi_len": csi_len,
        "flags": flags,
        "queue_drops": queue_drops,
        "csi": csi,
    }


# ============================================================
# OUTPUT FILE
# ============================================================

def next_filename(mode):
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    index = 1

    while True:

        filename = os.path.join(
            OUTPUT_DIR,
            f"{mode}_{index:02d}.csv"
        )

        if not os.path.exists(filename):
            return filename

        index += 1


# ============================================================
# MAIN
# ============================================================

if len(sys.argv) != 2:

    print(
        "\nUsage:\n"
        "  python collect.py idle\n"
        "  python collect.py motion\n"
    )

    sys.exit(1)


mode = sys.argv[1].lower()

if mode not in ("idle", "motion"):

    print(
        "[ERROR] Mode must be "
        "'idle' or 'motion'."
    )

    sys.exit(1)


label = 0 if mode == "idle" else 1

output_file = next_filename(mode)


print("=" * 65)
print("RF VISION — CSI DATA COLLECTOR")
print("=" * 65)

print(f"Mode              : {mode.upper()}")
print(f"Label             : {label}")
print(f"Port              : {PORT}")
print(f"Baud              : {BAUD}")
print(
    f"Duration          : "
    f"{RECORD_DURATION_SEC} s"
)

print(
    f"Processing rate   : "
    f"{PROCESSING_RATE_HZ} Hz"
)

print(f"Output            : {output_file}")

print()
print(
    "IMPORTANT: Keep the environment in the "
    f"{mode.upper()} state."
)

print(
    "Starting in 5 seconds..."
)

for i in range(5, 0, -1):

    print(i)

    time.sleep(1)

print()
print("RECORDING STARTED")
print()


# ------------------------------------------------------------
# SERIAL
# ------------------------------------------------------------

try:

    ser = serial.Serial(
        PORT,
        BAUD,
        timeout=0.2
    )

    ser.dtr = False
    ser.rts = False

except Exception as e:

    print(
        f"[ERROR] Could not open {PORT}: {e}"
    )

    sys.exit(1)


# ------------------------------------------------------------
# RECORD
# ------------------------------------------------------------

start_wall = time.time()

packet_count = 0
crc_or_parse_errors = 0

timestamps = []

max_queue_drops = 0

rows = []


while (
    time.time() - start_wall
    < RECORD_DURATION_SEC
):

    packet = read_csi_packet(ser)

    if packet is None:

        crc_or_parse_errors += 1
        continue

    packet_count += 1

    timestamp_sec = (
        packet["timestamp_us"] /
        1_000_000.0
    )

    timestamps.append(
        timestamp_sec
    )

    max_queue_drops = max(
        max_queue_drops,
        packet["queue_drops"]
    )

    csi_hex = packet["csi"].hex()

    rows.append([
        packet["sequence"],
        packet["timestamp_us"],
        packet["rssi"],
        packet["channel"],
        packet["csi_len"],
        packet["flags"],
        packet["queue_drops"],
        label,
        csi_hex,
    ])

    if packet_count % 100 == 0:

        elapsed = (
            time.time() - start_wall
        )

        rate = (
            packet_count / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"[{elapsed:6.1f}s] "
            f"packets={packet_count:5d} "
            f"rate={rate:5.2f} Hz"
        )


ser.close()


# ============================================================
# SAVE
# ============================================================

with open(
    output_file,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "sequence",
        "timestamp_us",
        "rssi",
        "channel",
        "csi_len",
        "flags",
        "queue_drops",
        "label",
        "csi_hex",
    ])

    writer.writerows(rows)


# ============================================================
# STATISTICS
# ============================================================

print()
print("=" * 65)
print("RECORDING COMPLETE")
print("=" * 65)

print(
    f"Packets recorded : {packet_count}"
)

print(
    f"Parse errors     : {crc_or_parse_errors}"
)

print(
    f"Max queue drops  : {max_queue_drops}"
)

if len(timestamps) >= 2:

    timestamps = np.asarray(
        timestamps,
        dtype=np.float64
    )

    intervals = np.diff(
        timestamps
    )

    median_dt = np.median(
        intervals
    )

    effective_rate = (
        1.0 /
        np.mean(intervals)
    )

    print(
        f"Median interval  : "
        f"{median_dt * 1000:.3f} ms"
    )

    print(
        f"Effective rate   : "
        f"{effective_rate:.3f} Hz"
    )

    print(
        f"Recommended rate : "
        f"{round(effective_rate)} Hz"
    )

print()
print(
    f"Saved to:\n{output_file}"
)
print()