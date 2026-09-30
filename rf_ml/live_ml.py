"""
RF Vision — LIVE CSI MOTION DETECTOR

ESP32 CSI
    ↓
timestamped CSI packets
    ↓
selected CSI amplitudes
    ↓
timestamp-aware 8 Hz resampling
    ↓
3-second / 24-sample windows
    ↓
44 features
    ↓
Random Forest
    ↓
temporal voting
    ↓
IDLE / MOTION
"""

import time
import struct
import json
from collections import deque

import joblib
import numpy as np
import serial
import matplotlib.pyplot as plt

from csi_features import (
    SAMPLE_RATE_HZ,
    WINDOW_SECONDS,
    WINDOW_SAMPLES,
    CSI_PAIR_INDICES,
    FEATURE_NAMES,
    extract_selected_amplitudes,
    build_feature_windows,
    extract_feature_vector,
)


# ============================================================
# CONFIGURATION
# ============================================================

PORT = "COM3"
BAUD = 115200

MODEL_FILE = "rf_model.joblib"
CONFIG_FILE = "model_config.json"

MAGIC = b"\xA5\x5A"

MAX_CSI_LEN = 256
HEADER_AFTER_MAGIC = 22

CALIBRATION_SECONDS = 5

PREDICTION_INTERVAL = 0.5

PREDICTION_HISTORY = 7
MOTION_VOTE_THRESHOLD = 4

HISTORY_SECONDS = 10

RAW_HISTORY_SIZE = 600

DOPPLER_MAX_HZ = 3.5


# ============================================================
# STARTUP
# ============================================================

print("=" * 70)
print("RF VISION — LIVE CSI MOTION DETECTOR")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load(MODEL_FILE)
except Exception as e:
    raise SystemExit(
        f"[ERROR] Could not load {MODEL_FILE}: {e}"
    )


# ============================================================
# LOAD MODEL CONFIG
# ============================================================

try:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        model_config = json.load(f)
except Exception as e:
    raise SystemExit(
        f"[ERROR] Could not load {CONFIG_FILE}: {e}"
    )


# ============================================================
# VERIFY MODEL CONFIGURATION
# ============================================================

checks = [
    (
        "sample rate",
        model_config.get("sample_rate_hz"),
        SAMPLE_RATE_HZ,
    ),
    (
        "window duration",
        model_config.get("window_seconds"),
        WINDOW_SECONDS,
    ),
    (
        "window samples",
        model_config.get("window_samples"),
        WINDOW_SAMPLES,
    ),
    (
        "CSI pair indices",
        model_config.get("csi_pair_indices"),
        CSI_PAIR_INDICES,
    ),
    (
        "feature names",
        model_config.get("feature_names"),
        FEATURE_NAMES,
    ),
]

for name, saved, current in checks:

    if saved != current:

        raise SystemExit(
            f"[ERROR] Model {name} does not match "
            f"csi_features.py.\n"
            f"Saved  : {saved}\n"
            f"Current: {current}"
        )


print()
print("MODEL")
print("-" * 70)
print(f"Sample rate : {SAMPLE_RATE_HZ} Hz")
print(f"Window      : {WINDOW_SECONDS:.1f} s")
print(f"Samples     : {WINDOW_SAMPLES}")
print(f"CSI pairs   : {CSI_PAIR_INDICES}")
print(f"Features    : {len(FEATURE_NAMES)}")
print(f"Classes     : {model.classes_}")

if hasattr(model, "n_estimators"):
    print(f"Trees       : {model.n_estimators}")


# ============================================================
# CLASS CHECK
# ============================================================

classes = list(model.classes_)

if 0 not in classes or 1 not in classes:

    raise SystemExit(
        "[ERROR] Model must contain both "
        "IDLE=0 and MOTION=1 classes."
    )

MOTION_CLASS_INDEX = classes.index(1)


# ============================================================
# CRC
# ============================================================

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


# ============================================================
# SERIAL READ
# ============================================================

def read_exact(ser, n):

    data = bytearray()

    while len(data) < n:

        chunk = ser.read(n - len(data))

        if not chunk:
            return None

        data.extend(chunk)

    return bytes(data)


# ============================================================
# READ CSI PACKET
# ============================================================

def read_csi_packet(ser):

    while True:

        first = ser.read(1)

        if not first:
            return None

        if first != b"\xA5":
            continue

        second = ser.read(1)

        if not second:
            return None

        if second == b"\x5A":
            break

    header = read_exact(
        ser,
        HEADER_AFTER_MAGIC
    )

    if header is None:
        return None

    try:

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

    except struct.error:

        return None

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

    calculated_crc = crc16_ccitt(
        MAGIC +
        header +
        csi
    )

    if calculated_crc != received_crc:

        return None

    return {
        "sequence": sequence,
        "timestamp": timestamp_us / 1_000_000.0,
        "rssi": rssi,
        "channel": channel,
        "flags": flags,
        "queue_drops": queue_drops,
        "csi": csi,
    }


# ============================================================
# OPEN SERIAL
# ============================================================

try:

    ser = serial.Serial(
        PORT,
        BAUD,
        timeout=0.5
    )

    ser.dtr = False
    ser.rts = False

except Exception as e:

    raise SystemExit(
        f"[ERROR] Cannot open {PORT}: {e}"
    )


# ============================================================
# LIVE BUFFERS
# ============================================================

timestamps = deque(
    maxlen=RAW_HISTORY_SIZE
)

amplitude_history = deque(
    maxlen=RAW_HISTORY_SIZE
)

rssi_history = deque(
    maxlen=RAW_HISTORY_SIZE
)

prediction_history = deque(
    maxlen=PREDICTION_HISTORY
)

probability_history = deque(
    maxlen=PREDICTION_HISTORY
)


# ============================================================
# CALIBRATION
# ============================================================

print()
print(
    f"Collecting {CALIBRATION_SECONDS}s of CSI..."
)
print("Keep the scene NORMAL during startup.")

calibration_start = time.monotonic()
calibration_count = 0
calibration_last_seq = None
calibration_drops = 0

while (
    time.monotonic() -
    calibration_start <
    CALIBRATION_SECONDS
):

    packet = read_csi_packet(ser)

    if packet is None:
        continue

    calibration_count += 1

    calibration_last_seq = packet["sequence"]
    calibration_drops = packet["queue_drops"]


print(
    f"Calibration packets : "
    f"{calibration_count}"
)

print(
    f"Queue drops reported : "
    f"{calibration_drops}"
)

if calibration_count < 20:

    ser.close()

    raise SystemExit(
        "[ERROR] Too few CSI packets received."
    )


# Clear everything collected during calibration.

ser.reset_input_buffer()

timestamps.clear()
amplitude_history.clear()
rssi_history.clear()

prediction_history.clear()
probability_history.clear()


# ============================================================
# PLOT
# ============================================================

plt.ion()

fig = plt.figure(
    figsize=(12, 7)
)

ax_signal = fig.add_subplot(211)
ax_doppler = fig.add_subplot(212)

signal_line, = ax_signal.plot([], [])

ax_signal.set_title(
    "Mean CSI Amplitude"
)

ax_signal.set_xlabel(
    "Time (s)"
)

ax_signal.set_ylabel(
    "Amplitude"
)

ax_doppler.set_title(
    "Doppler Spectrum"
)

ax_doppler.set_xlabel(
    "Frequency (Hz)"
)

ax_doppler.set_ylabel(
    "Magnitude"
)

status_text = fig.suptitle(
    "WAITING FOR CSI...",
    fontsize=18,
    fontweight="bold",
    y=0.96
)

plt.subplots_adjust(
    top=0.88,
    hspace=0.45
)


# ============================================================
# STATE
# ============================================================

last_prediction_time = 0.0

last_rssi = 0

last_queue_drops = 0

last_sequence = None

packet_count = 0

valid_window_count = 0

last_features = None


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("LIVE DETECTION STARTED")
print("=" * 70)

print(
    f"Processing rate : {SAMPLE_RATE_HZ} Hz"
)

print(
    f"Window          : "
    f"{WINDOW_SECONDS:.1f} s / "
    f"{WINDOW_SAMPLES} samples"
)

print(
    f"Prediction      : "
    f"every {PREDICTION_INTERVAL:.1f} s"
)

print()
print("Keep the ESP32 and receiver stationary.")
print("Move normally in front of the sensing region.")
print("Press Ctrl+C to stop.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        packet = read_csi_packet(ser)

        if packet is None:
            continue

        packet_count += 1

        last_rssi = packet["rssi"]

        last_queue_drops = packet["queue_drops"]

        last_sequence = packet["sequence"]

        amplitudes = extract_selected_amplitudes(
            packet["csi"]
        )

        if not np.all(
            np.isfinite(amplitudes)
        ):
            continue

        timestamps.append(
            packet["timestamp"]
        )

        amplitude_history.append(
            amplitudes
        )

        rssi_history.append(
            packet["rssi"]
        )


        # ----------------------------------------------------
        # WAIT FOR ENOUGH RAW HISTORY
        # ----------------------------------------------------

        if len(timestamps) < WINDOW_SAMPLES:

            status_text.set_text(
                f"COLLECTING WINDOW  "
                f"{len(timestamps)}/"
                f"{WINDOW_SAMPLES}"
            )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            continue


        # ----------------------------------------------------
        # ARRAYS
        # ----------------------------------------------------

        t = np.asarray(
            timestamps,
            dtype=np.float64
        )

        a = np.asarray(
            amplitude_history,
            dtype=np.float64
        )


        # ----------------------------------------------------
        # BUILD FEATURES
        #
        # IMPORTANT:
        # build_feature_windows() returns:
        #
        #     X, y, window_times
        #
        # NOT a list of raw windows.
        # ----------------------------------------------------

        try:

            X_live, _, window_times = (
                build_feature_windows(
                    t,
                    a,
                    label=None
                )
            )

        except Exception as exc:

            status_text.set_text(
                "WINDOW PROCESSING ERROR"
            )

            print(
                f"\n[WINDOW ERROR] {exc}"
            )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            continue


        # ----------------------------------------------------
        # NO VALID WINDOW
        # ----------------------------------------------------

        if X_live.shape[0] == 0:

            status_text.set_text(
                "WAITING FOR CLEAN 3s WINDOW"
            )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            continue


        # ----------------------------------------------------
        # MOST RECENT FEATURE VECTOR
        # ----------------------------------------------------

        features = X_live[-1]

        valid_window_count += 1

        last_features = features


        # ----------------------------------------------------
        # FEATURE DIMENSION CHECK
        # ----------------------------------------------------

        if len(features) != len(FEATURE_NAMES):

            status_text.set_text(
                "FEATURE DIMENSION ERROR"
            )

            continue


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        now = time.monotonic()

        if (
            now -
            last_prediction_time
            >= PREDICTION_INTERVAL
        ):

            last_prediction_time = now

            X = features.reshape(
                1,
                -1
            )

            prediction = int(
                model.predict(X)[0]
            )

            probabilities = (
                model.predict_proba(X)[0]
            )

            motion_probability = float(
                probabilities[
                    MOTION_CLASS_INDEX
                ]
            )

            prediction_history.append(
                prediction
            )

            probability_history.append(
                motion_probability
            )

            motion_votes = sum(
                prediction_history
            )

            # ----------------------------------------------
            # TEMPORAL DECISION
            # ----------------------------------------------

            if (
                len(prediction_history) >= 3
                and
                motion_votes >=
                MOTION_VOTE_THRESHOLD
            ):

                state = "MOTION"

            else:

                state = "IDLE"


            # ----------------------------------------------
            # TERMINAL
            # ----------------------------------------------

            print(
                f"\r"
                f"STATE: {state:<6} | "
                f"MOTION: "
                f"{motion_probability:.3f} | "
                f"VOTES: "
                f"{motion_votes}/"
                f"{len(prediction_history)} | "
                f"RSSI: "
                f"{last_rssi:>4} dBm | "
                f"QDROP: "
                f"{last_queue_drops}",
                end="",
                flush=True
            )


            # ----------------------------------------------
            # GUI STATUS
            # ----------------------------------------------

            status_text.set_text(
                f"{state}   |   "
                f"Motion probability: "
                f"{motion_probability:.3f}   |   "
                f"RSSI: {last_rssi} dBm"
            )


        # ====================================================
        # SIGNAL PLOT
        # ====================================================

        if len(t) >= 2:

            display_signal = np.mean(
                a,
                axis=1
            )

            relative_time = (
                t - t[-1]
            )

            signal_line.set_data(
                relative_time,
                display_signal
            )

            ax_signal.set_xlim(
                -HISTORY_SECONDS,
                0
            )

            ymin = np.min(
                display_signal
            )

            ymax = np.max(
                display_signal
            )

            if ymax > ymin:

                margin = (
                    ymax - ymin
                ) * 0.10

                ax_signal.set_ylim(
                    ymin - margin,
                    ymax + margin
                )


        # ====================================================
        # DOPPLER
        # ====================================================

        if len(a) >= WINDOW_SAMPLES:

            display_for_fft = a[-WINDOW_SAMPLES:]

            spectrum_signal = np.mean(
                display_for_fft,
                axis=1
            )

            spectrum_signal = (
                spectrum_signal -
                np.mean(spectrum_signal)
            )

            # Hann window reduces spectral leakage.

            hann = np.hanning(
                len(spectrum_signal)
            )

            spectrum_signal = (
                spectrum_signal *
                hann
            )

            fft = np.abs(
                np.fft.rfft(
                    spectrum_signal
                )
            )

            freqs = np.fft.rfftfreq(
                len(spectrum_signal),
                d=1.0 / SAMPLE_RATE_HZ
            )

            mask = (
                freqs <= DOPPLER_MAX_HZ
            )

            ax_doppler.clear()

            ax_doppler.plot(
                freqs[mask],
                fft[mask]
            )

            ax_doppler.set_title(
                "Doppler Spectrum"
            )

            ax_doppler.set_xlabel(
                "Frequency (Hz)"
            )

            ax_doppler.set_ylabel(
                "Magnitude"
            )

            ax_doppler.set_xlim(
                0,
                DOPPLER_MAX_HZ
            )


        # ====================================================
        # GUI
        # ====================================================

        fig.canvas.draw_idle()
        fig.canvas.flush_events()


# ============================================================
# STOP
# ============================================================

except KeyboardInterrupt:

    print()
    print()
    print("Stopping live detector...")


finally:

    try:
        ser.close()
    except Exception:
        pass

    plt.ioff()

    try:
        plt.close(fig)
    except Exception:
        pass

    print(
        "Serial connection closed."
    )