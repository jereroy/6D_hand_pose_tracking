import socket
import time

UDP_IP = "127.0.0.1"   # Adresse locale
UDP_PORT = 5005        # Port où Unity écoute

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    message = "Hello from Python".encode('utf-8')
    sock.sendto(message, (UDP_IP, UDP_PORT))
    print("Sent:", message)
    time.sleep(1)
