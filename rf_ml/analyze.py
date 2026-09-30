import os

INPUT_FILE = "raw_csi.bin"

if not os.path.exists(INPUT_FILE):
    print(f"[!] File '{INPUT_FILE}' not found! Run capture.py first.")
    exit(1)

with open(INPUT_FILE, "rb") as f:
    data = f.read()

print("==============================================")
print("       RAW CSI BINARY STREAM ANALYSIS         ")
print("==============================================")
print(f"Total Stream Size     : {len(data)} bytes")

# 1. Count CSI Markers
marker_count = data.count(b"CSI,")
print(f"CSI Markers Found     : {marker_count}")

# 2. Analyze Individual Packets
text_stream = data.replace(b"\x00", b"").decode(errors="ignore")
lines = text_stream.split("\n")

valid_packets = 0
malformed_packets = 0

for line in lines:
    if "CSI," not in line:
        continue
    
    idx = line.find("CSI,")
    clean = line[idx:].strip()
    parts = clean.split(",")
    
    if len(parts) >= 5:
        try:
            declared_len = int(parts[4])
            payload = parts[5:5 + declared_len]
            if len(payload) == declared_len:
                # Verify payload numbers
                for v in payload:
                    float(v)
                valid_packets += 1
            else:
                malformed_packets += 1
        except (ValueError, IndexError):
            malformed_packets += 1
    else:
        malformed_packets += 1

print(f"Valid Complete Packets: {valid_packets}")
print(f"Malformed Packets     : {malformed_packets}")

if marker_count > 0:
    success_rate = (valid_packets / marker_count) * 100
    print(f"Packet Completion Rate: {success_rate:.1f}%")
print("==============================================")