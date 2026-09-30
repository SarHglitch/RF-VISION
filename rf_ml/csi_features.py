"""
RF Vision — Shared CSI Feature Extraction

Used by:
    collect.py
    train_model.py
    live_ml.py

IMPORTANT:
The CSI pair index is an ARRAY POSITION, not a physical Wi-Fi
subcarrier number.

ESP32 CSI stores each complex value as:
    imaginary byte, real byte

The current acquisition is approximately 7.8 Hz, therefore
the processing grid is locked to 8 Hz.
"""

import numpy as np
from scipy.signal import medfilt


# ============================================================
# PROCESSING CONFIGURATION
# ============================================================

SAMPLE_RATE_HZ = 8

WINDOW_SECONDS = 3.0
WINDOW_SAMPLES = int(SAMPLE_RATE_HZ * WINDOW_SECONDS)

STEP_SECONDS = 1.0
STEP_SAMPLES = int(SAMPLE_RATE_HZ * STEP_SECONDS)

# Do not interpolate through gaps this large.
MAX_INTERPOLATION_GAP_SEC = 0.50

# Human-motion Doppler region.
DOPPLER_BAND_HZ = (0.5, 3.0)

# Array positions in the CSI payload.
#
# Each pair is:
#   byte[2*i]     = imaginary
#   byte[2*i + 1] = real
#
# We deliberately call these PAIR INDICES rather than
# physical subcarrier numbers.
CSI_PAIR_INDICES = [
    5,
    6,
    7,
    8,
    9,
    10,
    11,
]


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = []

for i in CSI_PAIR_INDICES:
    FEATURE_NAMES.extend([
        f"pair_{i}_ptp",
        f"pair_{i}_iqr",
        f"pair_{i}_mad",
        f"pair_{i}_std",
        f"pair_{i}_diff_mad",
        f"pair_{i}_doppler",
    ])

FEATURE_NAMES.extend([
    "mean_cross_pair_std",
    "mean_cross_pair_range",
])


# ============================================================
# CSI BYTE DECODING
# ============================================================

def csi_to_amplitudes(csi_bytes):
    """
    Convert raw ESP32 CSI I/Q bytes into amplitudes.

    Each complex CSI value is represented as:

        imaginary, real

    with signed 8-bit I/Q values.
    """

    data = np.frombuffer(
        bytes(csi_bytes),
        dtype=np.int8
    ).astype(np.int16)

    usable = (
        len(data) // 2
    ) * 2

    if usable < 2:
        return np.empty(
            0,
            dtype=np.float64
        )

    data = data[:usable]

    imag = data[0::2]
    real = data[1::2]

    amplitudes = np.sqrt(
        imag.astype(np.float64) ** 2 +
        real.astype(np.float64) ** 2
    )

    return amplitudes

def extract_selected_amplitudes(csi_bytes):
    """
    Extract the selected CSI pair-index amplitudes.
    """

    amplitudes = csi_to_amplitudes(csi_bytes)

    values = []

    for pair_index in CSI_PAIR_INDICES:

        if pair_index >= len(amplitudes):
            values.append(np.nan)
        else:
            values.append(float(amplitudes[pair_index]))

    return np.asarray(values, dtype=np.float64)


# ============================================================
# RESAMPLING
# ============================================================

def robust_resample(
    timestamps,
    values,
    target_rate=SAMPLE_RATE_HZ,
    max_gap=MAX_INTERPOLATION_GAP_SEC,
):
    """
    Resample irregular CSI samples onto a uniform time grid.

    CRITICAL:
    We do NOT interpolate across long missing intervals.

    Long gaps become NaN.
    """

    timestamps = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    if len(timestamps) < 2:
        return None, None

    valid = (
        np.isfinite(timestamps) &
        np.isfinite(values)
    )

    timestamps = timestamps[valid]
    values = values[valid]

    if len(timestamps) < 2:
        return None, None

    order = np.argsort(timestamps)

    timestamps = timestamps[order]
    values = values[order]

    # Remove duplicate timestamps.
    unique_t, unique_idx = np.unique(
        timestamps,
        return_index=True
    )

    timestamps = unique_t
    values = values[unique_idx]

    if len(timestamps) < 2:
        return None, None

    period = 1.0 / target_rate

    grid = np.arange(
        timestamps[0],
        timestamps[-1] + period * 0.5,
        period,
    )

    if len(grid) < 2:
        return None, None

    result = np.full(
        len(grid),
        np.nan,
        dtype=np.float64
    )

    # Normal interpolation.
    interpolated = np.interp(
        grid,
        timestamps,
        values,
    )

    result[:] = interpolated

    # Find gaps in original samples.
    original_gaps = np.diff(timestamps)

    for i, gap in enumerate(original_gaps):

        if gap > max_gap:

            gap_start = timestamps[i]
            gap_end = timestamps[i + 1]

            mask = (
                (grid > gap_start) &
                (grid < gap_end)
            )

            result[mask] = np.nan

    return grid, result


# ============================================================
# WINDOW VALIDITY
# ============================================================

def window_is_valid(
    timestamps,
    values,
    minimum_valid_fraction=0.80,
):
    """
    Reject windows containing too much missing data.
    """

    values = np.asarray(values)

    if len(values) == 0:
        return False

    valid_fraction = np.mean(np.isfinite(values))

    return valid_fraction >= minimum_valid_fraction


# ============================================================
# SIGNAL FEATURES
# ============================================================

def _safe_median_filter(signal):
    """
    Apply a median filter while handling short windows.
    """

    signal = np.asarray(signal, dtype=np.float64)

    if len(signal) < 5:
        return signal.copy()

    return medfilt(signal, kernel_size=5)


def _single_signal_features(signal):
    """
    Extract six robust features from one CSI amplitude stream.
    """

    signal = np.asarray(signal, dtype=np.float64)

    if not np.all(np.isfinite(signal)):
        return None

    # Median filtering.
    filtered = _safe_median_filter(signal)

    # Small smoothing operation.
    if len(filtered) >= 3:
        smooth = np.convolve(
            filtered,
            np.ones(3) / 3.0,
            mode="same"
        )
    else:
        smooth = filtered

    # Remove static baseline.
    detrended = smooth - np.median(smooth)

    # Robust amplitude statistics.
    p95 = np.percentile(detrended, 95)
    p05 = np.percentile(detrended, 5)

    p75 = np.percentile(detrended, 75)
    p25 = np.percentile(detrended, 25)

    ptp_robust = p95 - p05
    iqr = p75 - p25

    mad = np.median(
        np.abs(
            detrended -
            np.median(detrended)
        )
    )

    lo, hi = np.percentile(
        detrended,
        [5, 95]
    )

    trimmed = detrended[
        (detrended >= lo) &
        (detrended <= hi)
    ]

    if len(trimmed) > 1:
        std_trimmed = np.std(trimmed)
    else:
        std_trimmed = 0.0

    if len(detrended) > 1:

        diff_mad = np.median(
            np.abs(
                np.diff(detrended)
            )
        )

    else:
        diff_mad = 0.0

    # Doppler energy.
    fft_mag = np.abs(
        np.fft.rfft(detrended)
    )

    freqs = np.fft.rfftfreq(
        len(detrended),
        d=1.0 / SAMPLE_RATE_HZ
    )

    total_energy = (
        np.sum(fft_mag ** 2) +
        1e-12
    )

    band_mask = (
        (freqs >= DOPPLER_BAND_HZ[0]) &
        (freqs <= DOPPLER_BAND_HZ[1])
    )

    doppler_energy = (
        np.sum(fft_mag[band_mask] ** 2) /
        total_energy
    )

    return [
        float(ptp_robust),
        float(iqr),
        float(mad),
        float(std_trimmed),
        float(diff_mad),
        float(doppler_energy),
    ]


# ============================================================
# WINDOW FEATURE EXTRACTION
# ============================================================

def extract_feature_vector(amplitude_window):
    """
    amplitude_window shape:

        [WINDOW_SAMPLES, number_of_selected_pairs]

    Returns:
        1-D feature vector
    """

    window = np.asarray(
        amplitude_window,
        dtype=np.float64
    )

    if window.ndim != 2:
        raise ValueError(
            "Expected 2-D amplitude window"
        )

    if window.shape[0] != WINDOW_SAMPLES:
        raise ValueError(
            f"Expected {WINDOW_SAMPLES} samples, "
            f"got {window.shape[0]}"
        )

    if window.shape[1] != len(CSI_PAIR_INDICES):
        raise ValueError(
            "Unexpected number of CSI pair streams"
        )

    features = []

    # --------------------------------------------------------
    # Per-pair features
    # --------------------------------------------------------

    for column in range(window.shape[1]):

        signal = window[:, column]

        if not np.all(np.isfinite(signal)):
            return None

        f = _single_signal_features(signal)

        if f is None:
            return None

        features.extend(f)

    # --------------------------------------------------------
    # Cross-pair consistency
    # --------------------------------------------------------

    cross_std = np.std(
        window,
        axis=1
    )

    cross_range = (
        np.max(window, axis=1) -
        np.min(window, axis=1)
    )

    features.append(
        float(np.mean(cross_std))
    )

    features.append(
        float(np.mean(cross_range))
    )

    return np.asarray(
        features,
        dtype=np.float64
    )


# ============================================================
# COMPLETE SESSION → FEATURE WINDOWS
# ============================================================

def build_feature_windows(
    timestamps,
    amplitude_matrix,
    label=None,
):
    """
    Convert one recorded session into ML windows.

    timestamps:
        shape [N]

    amplitude_matrix:
        shape [N, number_of_pairs]

    Returns:
        X, y, window_times
    """

    timestamps = np.asarray(
        timestamps,
        dtype=np.float64
    )

    amplitude_matrix = np.asarray(
        amplitude_matrix,
        dtype=np.float64
    )

    if len(timestamps) != len(amplitude_matrix):
        raise ValueError(
            "Timestamp and amplitude lengths differ"
        )

    if amplitude_matrix.ndim != 2:
        raise ValueError(
            "Amplitude matrix must be 2-D"
        )

    # Resample every pair.
    resampled = []

    common_grid = None

    for pair in range(
        amplitude_matrix.shape[1]
    ):

        grid, signal = robust_resample(
            timestamps,
            amplitude_matrix[:, pair]
        )

        if grid is None:
            return np.empty((0, len(FEATURE_NAMES))), [], []

        if common_grid is None:
            common_grid = grid

        resampled.append(signal)

    resampled = np.column_stack(resampled)

    X = []
    y = []
    window_times = []

    step = STEP_SAMPLES

    for start in range(
        0,
        len(common_grid) - WINDOW_SAMPLES + 1,
        step
    ):

        end = start + WINDOW_SAMPLES

        window = resampled[start:end]

        valid_fraction = np.mean(
            np.isfinite(window)
        )

        if valid_fraction < 0.80:
            continue

        # Fill tiny isolated missing values only.
        clean = window.copy()

        for col in range(clean.shape[1]):

            series = clean[:, col]

            if np.any(~np.isfinite(series)):

                good = np.isfinite(series)

                if np.sum(good) < 2:
                    break

                series[~good] = np.interp(
                    np.flatnonzero(~good),
                    np.flatnonzero(good),
                    series[good],
                )

                clean[:, col] = series

        else:

            feature_vector = extract_feature_vector(
                clean
            )

            if feature_vector is None:
                continue

            X.append(feature_vector)

            if label is not None:
                y.append(label)

            window_times.append(
                common_grid[start]
            )

    if not X:
        return (
            np.empty(
                (0, len(FEATURE_NAMES)),
                dtype=np.float64
            ),
            y,
            window_times,
        )

    return (
        np.vstack(X),
        y,
        window_times,
    )