"""
RF Vision — Live Doppler Spectrogram Interface
"""
import matplotlib
matplotlib.use('TkAgg')  # Force interactive Tkinter backend

import json
from collections import deque
import numpy as np
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import stft

# ---------------- EDIT THESE FOR YOUR SETUP ----------------
PORT = "COM3"
BAUD = 115200
SAMPLE_RATE_HZ = 100 
WINDOW_SEC = 4 
NPERSEG = 128 
# -------------------------------------------------------------

# Load baseline settings
with open("baseline.json") as f:
    baseline = json.load(f)

SUBCARRIER_IDX = baseline["subcarrier_idx"]
NOISE_FLOOR = baseline["noise_floor"]

BUF_LEN = SAMPLE_RATE_HZ * WINDOW_SEC
amp_buffer = deque([baseline.get("amp_mean", 15.0)] * BUF_LEN, maxlen=BUF_LEN)


def parse_line(line):
    parts = line.strip().split(",")
    if len(parts) < 5:
        return None
    try:
        start_idx = 0
        for i, p in enumerate(parts):
            if p.lstrip('-').isdigit():
                start_idx = i
                break
        
        if len(parts) < start_idx + 4:
            return None
            
        length = int(parts[start_idx + 3])
        raw_bytes = parts[start_idx + 4 : start_idx + 4 + length]
        return np.array(raw_bytes, dtype=np.int8)
    except (ValueError, Exception):
        return None


def csi_to_amp(csi_bytes, subcarrier_idx):
    idx = subcarrier_idx * 2
    if idx + 1 >= len(csi_bytes):
        return None
    imag = float(csi_bytes[idx])
    real = float(csi_bytes[idx + 1])
    amp = np.sqrt(real ** 2 + imag ** 2)
    return amp if amp > 0 else None


# Open Serial Port
ser = serial.Serial(PORT, BAUD, timeout=0.1)

# Initialize Figure and Subplots
fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(8, 6))
fig.suptitle("RF Vision — Live Doppler Output", fontsize=13, fontweight="bold")


def update(_frame):
    global amp_buffer
    
    # 1. Consume all available serial packets
    while ser.in_waiting:
        raw = ser.readline().decode(errors="ignore")
        csi = parse_line(raw)
        if csi is None:
            continue
            
        amp = csi_to_amp(csi, SUBCARRIER_IDX)
        if amp is not None:
            # maxlen=BUF_LEN automatically drops the oldest sample
            amp_buffer.append(amp)

    # 2. Process amplitude wave data
    sig = np.array(amp_buffer)
    motion = np.max(sig) > NOISE_FLOOR

    ax_wave.clear()
    ax_wave.plot(sig, color="#2b5c9e")
    ax_wave.axhline(NOISE_FLOOR, color="red", linestyle="--", linewidth=1, label="noise floor")
    ax_wave.set_title("MOTION DETECTED!" if motion else "idle (below noise floor)",
                     color="red" if motion else "gray", fontweight="bold")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.legend(loc="upper right", fontsize=8)
    ax_wave.set_ylim(0, max(NOISE_FLOOR * 1.5, float(np.max(sig)) * 1.1))

    # 3. Calculate and plot Short-Time Fourier Transform (Doppler Spectrogram)
    f, t, Zxx = stft(sig - np.mean(sig), fs=SAMPLE_RATE_HZ, nperseg=NPERSEG)
    ax_spec.clear()
    ax_spec.pcolormesh(t, f, np.abs(Zxx), shading="auto", cmap="inferno")
    ax_spec.set_ylabel("Doppler frequency (Hz)")
    ax_spec.set_xlabel("Time (s, rolling window)")
    ax_spec.set_title("Doppler Spectrogram")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])


ani = animation.FuncAnimation(fig, update, interval=50, cache_frame_data=False)
plt.show()