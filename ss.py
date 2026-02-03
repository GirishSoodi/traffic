import socket
import time

UDP_IP = "127.0.0.1"
UDP_PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("🚗 SUMO: Sending UDP packets to NS-3...")

while True:
    msg = "Hello from SUMO".encode()
    sock.sendto(msg, (UDP_IP, UDP_PORT))
    print("Sent:", msg)
    time.sleep(1)
