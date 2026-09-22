"""
RF Vision — Master Production Motion HUD
(Filtered Static Corridor [6.0 .. 29.0] + Fast Serial Drain + Fixed Spectrogram)
"""
import matplotlib
matplotlib.use('TkAgg')  # Fast interactive backend

import json
from collections import deque
import numpy as np
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import stft, medfilt
from scipy.ndimage import uniform_filter1d

# ---------------- MASTER CONFIGURATION ----------------
PORT = "COM3"
BAUD = 115200
SAMPLE_RATE_HZ = 100 
WINDOW_SEC = 4 

# STFT OPTIMIZATION (Zero UI Lag)
NPERSEG = 32        
NOVERLAP = 16

# 1. EXPANDED STATIC BOUNDS (Prevents false positives)
UPPER_BOUND = 29.0
LOWER_BOUND = 6.0

# 2. DISARM SWING THRESHOLD
RELEASE_SWING_THRESH = 5.5  # 150ms swing < 5.5 indicates idle state

# 3. FAST-RESPONSE HYSTERESIS COUNTERS
FRAMES_TO_ARM = 1           # Instant 1-frame trigger (~30ms) for zero delay
FRAMES_TO_DISARM = 3        # Fast 3-frame disarm (~90ms)
# ------------------------------------------------------

# Load baseline settings
with open("baseline.json") as f:
    baseline = json.load(f)

SUBCARRIER_IDX = baseline["subcarrier_idx"]
AMP_MEAN = baseline.get("amp_mean", 17.5)

BUF_LEN = SAMPLE_RATE_HZ * WINDOW_SEC
amp_buffer = deque([AMP_MEAN] * BUF_LEN, maxlen=BUF_LEN)

# State tracking variables
motion_state = False
above_count = 0
below_count = 0


def parse_line(line):
    parts = line.strip().split(",")
    if len(parts) < 6 or parts[0] != "CSI":
        return None
    try:
        length = int(parts[4])
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
    imag = float(csi_bytes[idx])
    real = float(csi_bytes[idx + 1])
    amp = np.sqrt(real ** 2 + imag ** 2)
    return amp if amp > 0 else None


# Open Serial Port
ser = serial.Serial()
ser.port = PORT
ser.baudrate = BAUD
ser.timeout = 0.005  # Minimal timeout for zero serial queue delay
ser.dtr = False
ser.rts = False
ser.open()

# --- Y2K / TECH-HUD STYLING SETUP ---
BG_COLOR = "#0A0A0A"
GRID_COLOR = "#222222"
TEXT_COLOR = "#EEEEEE"
ACCENT_GREEN = "#76BA1B"
ACCENT_PINK = "#FF5588"
WAVE_COLOR = "#00E5FF"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "font.family": "monospace"
})

fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(9, 7))
fig.suptitle(" [ RF_VISION // STATIC_CORRIDOR_HUD ] ", fontsize=13, fontweight="bold", color=TEXT_COLOR)
fig.subplots_adjust(hspace=0.40, top=0.90, bottom=0.10, left=0.10, right=0.95)


def setup_hud_axis(ax, title):
    ax.set_title(title, fontsize=10, loc="left", pad=8, fontweight="bold")
    ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(1.2)


setup_hud_axis(ax_wave, f">> AMPLITUDE_STREAM (BOUNDS: [{LOWER_BOUND:.1f} .. {UPPER_BOUND:.1f}])")
setup_hud_axis(ax_spec, ">> BIPOLAR_DOPPLER_SPECTROGRAM")

# Static Line Artists for zero-flicker rendering
wave_line, = ax_wave.plot(np.array(amp_buffer), color=WAVE_COLOR, linewidth=1.1, label="RAW_AMP")
upper_thresh_line = ax_wave.axhline(UPPER_BOUND, color=ACCENT_PINK, linestyle="--", linewidth=1.2, label=f"UPPER_BOUND ({UPPER_BOUND})")
lower_thresh_line = ax_wave.axhline(LOWER_BOUND, color=ACCENT_PINK, linestyle="--", linewidth=1.2, label=f"LOWER_BOUND ({LOWER_BOUND})")

status_badge = ax_wave.text(0.02, 0.84, " [*] SYSTEM_IDLE ", transform=ax_wave.transAxes,
                            color=BG_COLOR, bbox=dict(boxstyle="square,pad=0.3", facecolor=ACCENT_GREEN, edgecolor="none"),
                            fontweight="bold", fontsize=9)

ax_wave.set_ylabel("Amplitude", fontsize=8)
ax_wave.set_ylim(0, 35)  # Fixed view scale
ax_wave.legend(loc="upper right", facecolor=BG_COLOR, edgecolor="#444444", fontsize=7)

dummy_spec = np.zeros((17, 10))
# FIXED SPECTROGRAM COLOR SCALE (vmin=0.0, vmax=12.0) prevents wild background flashing
spec_img = ax_spec.imshow(dummy_spec, aspect='auto', origin='lower',
                          extent=[0, WINDOW_SEC, -8.0, 8.0], cmap='inferno', vmin=0.0, vmax=12.0)
ax_spec.axhline(0.0, color="#00E5FF", linestyle=":", linewidth=0.8, alpha=0.7)
ax_spec.set_ylabel("Doppler Shift (Hz)", fontsize=8)
ax_spec.set_xlabel("Time (s)", fontsize=8)


def update(_frame):
    global amp_buffer, motion_state, above_count, below_count
    
    # 1. ULTRA-FAST SERIAL DRAIN (Eliminates Lag)
    if ser.in_waiting > 1000:
        ser.reset_input_buffer()

    packets_read = 0
    while ser.in_waiting and packets_read < 40:
        raw = ser.readline().decode(errors="ignore")
        csi = parse_line(raw)
        if csi is None:
            continue
            
        amp = csi_to_amp(csi, SUBCARRIER_IDX)
        if amp is not None:
            amp_buffer.append(amp)
        packets_read += 1

    # 2. FAST OUTLIER DE-SPIKING (Filter single-sample noise spikes)
    sig_raw = np.array(amp_buffer)
    sig_med = medfilt(sig_raw, kernel_size=5)  # 5-point median strips spikes cleanly
    sig_smooth = uniform_filter1d(sig_med, size=3)

    # 3. FAST-RESPONSE TRIGGER & SWING RELEASE LOGIC
    recent_15 = sig_smooth[-15:]
    
    # Trigger: Check breakout against bounds [6.0 .. 29.0]
    amp_breakout = (np.max(recent_15) > UPPER_BOUND) or (np.min(recent_15) < LOWER_BOUND)
    
    # Release: Short 150ms swing settles below 5.5
    instant_swing = np.ptp(recent_15)
    motion_settled = instant_swing < RELEASE_SWING_THRESH

    if not motion_state:
        if amp_breakout:
            above_count += 1
            below_count = 0
        else:
            above_count = 0
    else:
        if motion_settled and not amp_breakout:
            below_count += 1
            above_count = 0
        else:
            below_count = 0

    if not motion_state and above_count >= FRAMES_TO_ARM:
        motion_state = True
    elif motion_state and below_count >= FRAMES_TO_DISARM:
        motion_state = False

    # 4. UI ARTIST UPDATES
    wave_line.set_ydata(sig_smooth)
    
    ax_wave.set_title(f">> AMPLITUDE_STREAM (BOUNDS: [{LOWER_BOUND:.1f} .. {UPPER_BOUND:.1f}] | SWING: {instant_swing:.1f})", 
                      fontsize=10, loc="left", pad=8, fontweight="bold")

    if motion_state:
        status_badge.set_text(" [!] MOTION_DETECTED ")
        status_badge.set_bbox(dict(boxstyle="square,pad=0.3", facecolor=ACCENT_PINK, edgecolor="none"))
    else:
        status_badge.set_text(" [*] SYSTEM_IDLE ")
        status_badge.set_bbox(dict(boxstyle="square,pad=0.3", facecolor=ACCENT_GREEN, edgecolor="none"))

    # 5. STABLE SPECTROGRAM RENDERING (Fixed Intensity Bounds)
    recent_sig = sig_smooth[-64:]
    sig_detrend = recent_sig - np.mean(recent_sig)
    
    f, t, Zxx = stft(sig_detrend, fs=SAMPLE_RATE_HZ, nperseg=NPERSEG, noverlap=NOVERLAP, return_onesided=False)
    f_shifted = np.fft.fftshift(f)
    spectrogram_mag = np.abs(np.fft.fftshift(Zxx, axes=0))

    freq_display_mask = (f_shifted >= -8.0) & (f_shifted <= 8.0)
    
    # Render with absolute spectral magnitude (stops auto-scaling color flashes)
    spec_img.set_data(spectrogram_mag[freq_display_mask, :])


try:
    ani = animation.FuncAnimation(fig, update, interval=25, cache_frame_data=False)
    plt.show()
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("\n[+] Serial port COM3 closed cleanly.")