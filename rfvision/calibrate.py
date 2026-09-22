"""
RF Vision — Baseline Calibration Script

Run this in an empty room (no human motion) for 30 seconds.
It collects raw CSI data across subcarriers, picks the subcarrier with the most
stable signal, calculates the mean amplitude and noise floor, and saves the
baseline configuration to `baseline.json`.
"""

import json
import time
from collections import defaultdict
import numpy as np
import serial

PORT = "COM3"
BAUD = 115200
CALIBRATION_DURATION_SEC = 30


def parse_line(line):
    parts = line.strip().split(",")
    # Needs CSI prefix + 4 header parts (timestamp, rssi, channel, len) + csi payload
    if len(parts) < 6 or parts[0] != "CSI":
        return None
    try:
        timestamp = int(parts[1])
        rssi = int(parts[2])
        channel = int(parts[3])
        length = int(parts[4])
        
        # Parse signed integers directly into Python floats
        raw_bytes = [float(x) for x in parts[5 : 5 + length]]
        if len(raw_bytes) < length:
            return None
            
        return np.array(raw_bytes, dtype=np.float32)
    except (ValueError, IndexError):
        return None


def csi_to_amp(csi_bytes, subcarrier_idx):
    idx = subcarrier_idx * 2
    if idx + 1 >= len(csi_bytes):
        return None
    imag = csi_bytes[idx]
    real = csi_bytes[idx + 1]
    amp = np.sqrt(real ** 2 + imag ** 2)
    return amp if amp > 0 else None


def main():
    print(f"Opening {PORT}...")
    ser = serial.Serial()
    ser.port = PORT
    ser.baudrate = BAUD
    ser.timeout = 0.1
    ser.dtr = False
    ser.rts = False
    ser.open()

    print(f"Reading from {PORT}. Capturing {CALIBRATION_DURATION_SEC}s of EMPTY-ROOM baseline...")
    print("Make sure nobody is in the detection zone right now.\n")

    subcarrier_samples = defaultdict(list)
    start_time = time.time()

    while time.time() - start_time < CALIBRATION_DURATION_SEC:
        if ser.in_waiting:
            raw = ser.readline().decode(errors="ignore")
            csi = parse_line(raw)
            if csi is None:
                continue

            num_subcarriers = len(csi) // 2
            for idx in range(num_subcarriers):
                amp = csi_to_amp(csi, idx)
                if amp is not None:
                    subcarrier_samples[idx].append(amp)

        elapsed = int(time.time() - start_time)
        print(f"\rCalibrating... {elapsed}/{CALIBRATION_DURATION_SEC}s", end="", flush=True)

    ser.close()
    print("\n\nData capture complete. Analyzing subcarrier stability...")

    if not subcarrier_samples:
        print("ERROR: No valid CSI packets captured. Check traffic generator and serial connection.")
        return

    # Pick the subcarrier with the lowest relative standard deviation (most stable in static room)
    best_idx = None
    best_score = float("inf")
    best_stats = None

    for idx, samples in subcarrier_samples.items():
        if len(samples) < 10:
            continue
        arr = np.array(samples)
        mean_amp = np.mean(arr)
        std_amp = np.std(arr)
        
        if mean_amp == 0:
            continue

        score = std_amp / mean_amp  # Coefficient of variation
        if score < best_score:
            best_score = score
            best_idx = idx
            best_stats = {
                "amp_mean": float(mean_amp),
                "amp_std": float(std_amp),
                "noise_floor": float(mean_amp + 3 * std_amp),  # 3-sigma noise floor threshold
                "n_samples": len(samples)
            }

    if best_idx is None:
        print("ERROR: Could not compute baseline statistics.")
        return

    result = {
        "subcarrier_idx": int(best_idx),
        "amp_mean": round(best_stats["amp_mean"], 2),
        "amp_std": round(best_stats["amp_std"], 2),
        "noise_floor": round(best_stats["noise_floor"], 2),
        "n_samples": best_stats["n_samples"]
    }

    with open("baseline.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n--- Baseline Results ---")
    print(f"Selected Subcarrier Index : {result['subcarrier_idx']}")
    print(f"Mean Amplitude            : {result['amp_mean']}")
    print(f"Standard Deviation        : {result['amp_std']}")
    print(f"Calculated Noise Floor    : {result['noise_floor']}")
    print(f"Samples Evaluated         : {result['n_samples']}")
    print("\nSaved to baseline.json successfully!")


if __name__ == "__main__":
    main()