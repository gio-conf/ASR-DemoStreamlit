import socket
import struct
import numpy as np
import torch

HOST = "0.0.0.0"
PORT = 50007

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)

print("WSL in ascolto...")
conn, addr = sock.accept()
print("Connesso a:", addr)

def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

audio_buffer = []

while True:
    # ricevi lunghezza
    header = recv_exact(conn, 4)
    if header is None:
        break
    length = struct.unpack(">I", header)[0]

    # ricevi audio
    data = recv_exact(conn, length)
    if data is None:
        break

    # converti in float32
    pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    audio_buffer.extend(pcm.tolist())
    print("ricevuti campioni:",len(pcm))
    # qui puoi chiamare il tuo stream_with_lookbehind_and_lookahead
    # esattamente come facevi con il file WAV
