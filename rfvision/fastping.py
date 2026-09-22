import socket
import time

# Update to match your room router's actual IP (e.g., 192.168.1.1 or 192.168.0.1)
ROUTER_IP = "192.168.0.1" 
PORT = 12345

# Create non-blocking UDP socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
payload = b"X" * 512  # 512-byte payload forces reliable 802.11 frame modulation

print(f"[+] Flooding high-frequency UDP packets to {ROUTER_IP}...")
print("[+] Keep this terminal running in the background while executing calibrate.py!")
print("[+] Press Ctrl+C to stop.\n")

packet_count = 0
start_time = time.time()

try:
    while True:
        # Send a burst of 10 packets per loop tick to bypass Windows sleep precision limits
        for _ in range(10):
            s.sendto(payload, (ROUTER_IP, PORT))
            packet_count += 1
        
        # Tiny 1ms pause to prevent local socket buffer saturation while maintaining 200+ Hz
        time.sleep(0.001)

        # Print live packet counter every 2 seconds
        if time.time() - start_time >= 2.0:
            rate = packet_count / (time.time() - start_time)
            print(f"\rTransmitting at: {rate:.1f} packets/sec | Total Sent: {packet_count}", end="", flush=True)

except KeyboardInterrupt:
    print("\n\n[-] Traffic generation stopped cleanly.")
    s.close()