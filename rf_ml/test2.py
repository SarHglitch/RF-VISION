import serial
import struct
import time
import statistics

from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

PORT = "COM3"
BAUD = 115200

TEST_DURATION = 30.0

MAGIC = b"\xA5\x5A"
PROTOCOL_VERSION = 1
MAX_CSI_LEN = 256


# ============================================================
# CRC16-CCITT
# ============================================================

def crc16_ccitt(data):

    crc = 0xFFFF

    for byte in data:

        crc ^= byte << 8

        for _ in range(8):

            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF

    return crc


# ============================================================
# READ EXACT NUMBER OF BYTES
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
# FIND PACKET HEADER
# ============================================================

def find_magic(ser):

    previous = None

    while True:

        byte = ser.read(1)

        if not byte:
            return None

        if previous == b"\xA5" and byte == b"\x5A":
            return MAGIC

        previous = byte


# ============================================================
# READ ONE CSI PACKET
# ============================================================

def read_packet(ser):

    magic = find_magic(ser)

    if magic is None:
        return None

    # New header:
    #
    # version       1
    # sequence      4
    # timestamp_us 8
    # RSSI          1
    # channel       1
    # CSI length    2
    # flags         1
    # queue drops   4
    #
    # TOTAL = 22 bytes

    header = read_exact(ser, 22)

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
        queue_drops
    ) = struct.unpack(
        "<BIQbBHBI",
        header
    )

    if version != PROTOCOL_VERSION:
        return None

    if csi_len <= 0 or csi_len > MAX_CSI_LEN:
        return None

    csi = read_exact(ser, csi_len)

    if csi is None:
        return None

    crc_bytes = read_exact(ser, 2)

    if crc_bytes is None:
        return None

    received_crc = struct.unpack(
        "<H",
        crc_bytes
    )[0]

    frame_without_crc = (
        MAGIC +
        header +
        csi
    )

    calculated_crc = crc16_ccitt(
        frame_without_crc
    )

    if calculated_crc != received_crc:

        return {
            "crc_error": True
        }

    return {

        "crc_error": False,

        "sequence": sequence,

        "timestamp_us": timestamp_us,

        "rssi": rssi,

        "channel": channel,

        "length": csi_len,

        "flags": flags,

        "queue_drops": queue_drops,

        "csi": csi
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 64)
    print("ESP32 CSI TIMING DIAGNOSTIC")
    print("=" * 64)

    print(f"Port             : {PORT}")
    print(f"Baud             : {BAUD}")
    print(f"Measurement time : {TEST_DURATION:.0f} s")

    print("=" * 64)
    print()

    ser = serial.Serial(
        PORT,
        BAUD,
        timeout=1
    )

    time.sleep(1)

    ser.reset_input_buffer()

    print("Listening for CSI...")
    print()

    packets = 0
    crc_errors = 0

    first_sequence = None
    last_sequence = None

    sequence_gaps = 0
    missing_sequences = 0

    max_queue_drops = 0

    timestamps_us = []
    rssi_values = []

    lengths = Counter()

    start = time.monotonic()

    # ========================================================
    # RECEIVE
    # ========================================================

    while time.monotonic() - start < TEST_DURATION:

        packet = read_packet(ser)

        if packet is None:
            continue

        if packet.get("crc_error", False):

            crc_errors += 1
            continue

        packets += 1

        sequence = packet["sequence"]
        timestamp_us = packet["timestamp_us"]

        timestamps_us.append(timestamp_us)

        rssi_values.append(packet["rssi"])

        lengths[packet["length"]] += 1

        max_queue_drops = max(
            max_queue_drops,
            packet["queue_drops"]
        )

        # ----------------------------------------------------
        # Sequence check
        # ----------------------------------------------------

        if first_sequence is None:

            first_sequence = sequence

        else:

            expected = last_sequence + 1

            if sequence > expected:

                gap = sequence - expected

                sequence_gaps += 1
                missing_sequences += gap

        last_sequence = sequence

    ser.close()

    actual_duration = time.monotonic() - start

    # ========================================================
    # CALCULATE INTERVALS
    # ========================================================

    intervals_ms = []

    for i in range(1, len(timestamps_us)):

        dt_us = (
            timestamps_us[i]
            - timestamps_us[i - 1]
        )

        if dt_us >= 0:

            intervals_ms.append(
                dt_us / 1000.0
            )

    # ========================================================
    # STATISTICS
    # ========================================================

    median_ms = None
    mean_ms = None
    p5_ms = None
    p95_ms = None

    effective_rate = None

    zero_or_near_zero = 0

    gaps_gt_250 = 0
    gaps_gt_500 = 0
    gaps_gt_1000 = 0

    rate_span = None

    if intervals_ms:

        median_ms = statistics.median(
            intervals_ms
        )

        mean_ms = statistics.mean(
            intervals_ms
        )

        sorted_intervals = sorted(
            intervals_ms
        )

        p5_ms = statistics.quantiles(
            sorted_intervals,
            n=100,
            method="inclusive"
        )[4]

        p95_ms = statistics.quantiles(
            sorted_intervals,
            n=100,
            method="inclusive"
        )[94]

        zero_or_near_zero = sum(
            dt < 1.0
            for dt in intervals_ms
        )

        gaps_gt_250 = sum(
            dt > 250
            for dt in intervals_ms
        )

        gaps_gt_500 = sum(
            dt > 500
            for dt in intervals_ms
        )

        gaps_gt_1000 = sum(
            dt > 1000
            for dt in intervals_ms
        )

        timestamp_span_us = (
            timestamps_us[-1]
            - timestamps_us[0]
        )

        rate_span = timestamp_span_us / 1_000_000.0

        if rate_span > 0:

            effective_rate = (
                (len(timestamps_us) - 1)
                / rate_span
            )

    # ========================================================
    # RECOMMENDED RATE
    # ========================================================

    recommended_rate = None

    if effective_rate is not None:

        recommended_rate = max(
            1,
            round(effective_rate)
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print("=" * 64)
    print("RESULT")
    print("=" * 64)

    print(
        f"Python test duration       : "
        f"{actual_duration:.2f} s"
    )

    print()
    print("PACKET INTEGRITY")
    print("-" * 64)

    print(
        f"Valid CSI packets          : "
        f"{packets}"
    )

    print(
        f"CRC errors                 : "
        f"{crc_errors}"
    )

    print()
    print("CSI LENGTH DISTRIBUTION")
    print("-" * 64)

    for length in sorted(lengths):

        count = lengths[length]

        percentage = (
            100.0 * count / packets
        )

        print(
            f"{length:4d} bytes : "
            f"{count:6d} packets "
            f"({percentage:6.2f}%)"
        )

    print()
    print("SEQUENCE INTEGRITY")
    print("-" * 64)

    print(
        f"First sequence             : "
        f"{first_sequence}"
    )

    print(
        f"Last sequence              : "
        f"{last_sequence}"
    )

    print(
        f"Sequence gaps              : "
        f"{sequence_gaps}"
    )

    print(
        f"Missing sequence packets  : "
        f"{missing_sequences}"
    )

    print()
    print("ESP32 QUEUE")
    print("-" * 64)

    print(
        f"Maximum queue drops        : "
        f"{max_queue_drops}"
    )

    print()
    print("TIMING ANALYSIS")
    print("-" * 64)

    if intervals_ms:

        print(
            f"Number of intervals        : "
            f"{len(intervals_ms)}"
        )

        print(
            f"Median interval            : "
            f"{median_ms:.3f} ms"
        )

        print(
            f"Mean interval              : "
            f"{mean_ms:.3f} ms"
        )

        print(
            f"5th percentile             : "
            f"{p5_ms:.3f} ms"
        )

        print(
            f"95th percentile            : "
            f"{p95_ms:.3f} ms"
        )

        print()
        print(
            f"Intervals < 1 ms           : "
            f"{zero_or_near_zero} "
            f"({100 * zero_or_near_zero / len(intervals_ms):.2f}%)"
        )

        print()
        print(
            f"Gaps > 250 ms              : "
            f"{gaps_gt_250} "
            f"({100 * gaps_gt_250 / len(intervals_ms):.2f}%)"
        )

        print(
            f"Gaps > 500 ms              : "
            f"{gaps_gt_500} "
            f"({100 * gaps_gt_500 / len(intervals_ms):.2f}%)"
        )

        print(
            f"Gaps > 1000 ms             : "
            f"{gaps_gt_1000} "
            f"({100 * gaps_gt_1000 / len(intervals_ms):.2f}%)"
        )

        print()
        print(
            f"ESP32 timestamp span       : "
            f"{rate_span:.3f} s"
        )

        print(
            f"Effective sample rate      : "
            f"{effective_rate:.3f} Hz"
        )

        print()
        print(
            f"RECOMMENDED PROCESSING RATE: "
            f"{recommended_rate} Hz"
        )

    else:

        print("Not enough data.")

    print()
    print("RSSI")
    print("-" * 64)

    if rssi_values:

        print(
            f"Average RSSI               : "
            f"{statistics.mean(rssi_values):.1f} dBm"
        )

        print(
            f"Minimum RSSI               : "
            f"{min(rssi_values)} dBm"
        )

        print(
            f"Maximum RSSI               : "
            f"{max(rssi_values)} dBm"
        )

    print()
    print("=" * 64)


if __name__ == "__main__":
    main()