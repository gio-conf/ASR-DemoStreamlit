import os
import sys
import re
from torch import nn, optim
import torchaudio
from torchaudio.models.decoder import ctc_decoder

from data import get_infer_data_loader
from models.model.early_exit import Early_conformer, full_conformer, Early_zipformer, Splitformer
from util.beam_infer import BeamInference
from util.conf import get_args
from util.data_loader import text_transform
from util.epoch_timer import epoch_time
from util.model_utils import *
from util.tokenizer import *
import torchaudio.transforms as T
import torch.nn.functional as F
import argparse
import numpy as np
import socket
import struct
import time
from collections import deque

def spec_transform(waveform, args):
    spec_t = T.Spectrogram(n_fft=args.n_fft * 2, hop_length=args.hop_length, win_length=args.win_length)
    return spec_t(waveform)

def melspec_transform(waveform, args):
    melspec_t = T.MelScale(sample_rate=args.sample_rate, n_mels=args.n_mels, n_stft=args.n_fft+1)
    return melspec_t(waveform)


#############################################
# PARAMETRI AUDIO E FINESTRE
#############################################

SAMPLE_RATE = 16000


LOOKBEHIND_SEC = 2.5 #0.5
CHUNK_SEC      = 0.51 #2.0
LOOKAHEAD_SEC  = 0.5 #0.5

LB = int(LOOKBEHIND_SEC * SAMPLE_RATE)
CK = int(CHUNK_SEC      * SAMPLE_RATE)
LA = int(LOOKAHEAD_SEC  * SAMPLE_RATE)

WINDOW = LB + CK + LA   # 4 secondi = 64000 campioni
ADVANCE = CK            # avanza di 2 secondi = 32000 campioni

# buffer PCM per accumulare voce
#pcm_buffer = deque(maxlen=SAMPLE_RATE * 60)  # 60 s max

# per segmentazione basata su heartbeat
SEGMENT_TIMEOUT = 2000 # 800 ms di soli heartbeat = fine segmento
FRAME_MS = 30
FRAME_TIMEOUT = SEGMENT_TIMEOUT / FRAME_MS

#############################################
# MODELLO E DECODER (DA INTEGRARE)
#############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Esempio: da sostituire con il tuo codice
# model = load_model(args).to(device).eval()
# decoder = StreamingCTCBeamSearchLexicon(...)

#############################################
# SOCKET
#############################################

HOST = "0.0.0.0"
PORT = 50007

'''
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)

print("WSL in ascolto...")
conn, addr = sock.accept()
print("Connesso a:", addr)
'''


_last_len = 0

def print_live_caption(partial):
    global _last_len

    # Normalizza l’output
    if isinstance(partial, list):
        text = " ".join(partial)      # <-- qui mettiamo gli spazi
    else:
        text = str(partial)
    sys.stdout.write("\x1b[2K\r")  # cancella la riga
    # Cancella la riga precedente
    sys.stdout.write("\r" + " " * _last_len)
    sys.stdout.write("\r" + text)
    sys.stdout.flush()

    _last_len = len(text)




def normalize_output(final):
    if isinstance(final, str):
        return final
    if isinstance(final, list):
        # se è lista di stringhe
        if all(isinstance(x, str) for x in final):
            return " ".join(final)
        # se è lista di tuple (word, score)
        if all(isinstance(x, tuple) for x in final):
            return " ".join(w for (w, s) in final)
        # fallback
        return " ".join(str(x) for x in final)
    # fallback generale
    return str(final)


def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

#############################################
# FUNZIONE: COSTRUISCI FINESTRA PER IL MODELLO
#############################################

def _build_window_from_buffer_(buffer):
    """
    Produce una finestra ogni CHUNK_SEC secondi:
    finestra = 1s LB + 2s CK + 1s LA = 4s totali
    Avanza il buffer di 2 secondi (CK).
    """

    # Se non abbiamo ancora 4 secondi -> niente finestra
    if len(buffer) < WINDOW:
        return None, buffer

    # Finestra completa
    window = buffer[:WINDOW]

    # Avanza di 2 secondi (chunk centrale)
    new_buffer = buffer[ADVANCE:]

    return window, new_buffer    

#############################################
# LOOP PRINCIPALE
#############################################

def build_window_from_buffer(buffer, final_flush=False):
    """
    buffer: numpy array float32 mono
    LB_SAMPLES: look-behind in samples
    CK_SAMPLES: central chunk in samples
    LA_SAMPLES: look-ahead in samples
    final_flush: True quando il VAD dice che ha finito lo speech
    """

    LB_SAMPLES = LB
    CK_SAMPLES = CK
    LA_SAMPLES = LA
    total_needed = LB_SAMPLES + CK_SAMPLES + LA_SAMPLES

    if final_flush:
        #if len(buffer) == 0:
        #if len(buffer) < total_needed:
        #    return None, buffer
        window = buffer.copy()
        new_buffer = buffer[:0]   # svuota
        buffer = buffer[len(buffer):]   # svuota        
        #return window, new_buffer
        return window, buffer

    # Se non abbiamo abbastanza per una finestra completa
    if len(buffer) < CK_SAMPLES:
        # troppo poco per emettere qualcosa
        #print("<CK_SAMPLES", final_flush)
        if not final_flush:
            return None, buffer
        # final flush: emetti tutto quello che c'e`
        window = buffer.copy()
        buffer = buffer[len(buffer):]   # svuota
        return window, buffer
        
    # Se abbiamo abbastanza per il chunk centrale
    # ma non abbastanza per LB o LA  usiamo quello che c’e`
    if len(buffer) < total_needed:
        #print("<total_need", final_flush)        
        if not final_flush:
            # aspettiamo ancora look-ahead
            return None, buffer

        # final flush: emetti finestra elastica
        # prendiamo tutto cio` che c’e`
        window = buffer.copy()
        buffer = buffer[len(buffer):]
        return window, buffer

    # Caso normale: finestra completa
    LB_ = buffer[:LB_SAMPLES]
    CK_ = buffer[LB_SAMPLES : LB_SAMPLES + CK_SAMPLES]
    LA_ = buffer[LB_SAMPLES + CK_SAMPLES : LB_SAMPLES + CK_SAMPLES + LA_SAMPLES]

    window = np.concatenate([LB_, CK_, LA_])

    # Avanza il buffer di CK (stride = 2 secondi)
    buffer = buffer[CK_SAMPLES:]

    return window, buffer




# parametri encoder (adatta ai tuoi)
SUBSAMPLING = 4
FEAT_FPS = 100  # feature frame per secondo prima del subsampling
EMB_PS = FEAT_FPS // SUBSAMPLING
LB_e = int(LOOKBEHIND_SEC * EMB_PS )
CK_e = int(CHUNK_SEC  * EMB_PS)

def handler_online(args, model, valid_len,  inf, dev, conn):
    
    print("Inizio streaming ASR...")
    raw_path = "central_stream.pcm"
    #central_f = open(raw_path, "wb")

    
    buffer = np.zeros(0, dtype=np.float32)
    count_silent_frames = 0 #

    while True:
        # ---- leggi header ----
        header = recv_exact(conn, 4)
        if header is None:
            break

        length = struct.unpack(">I", header)[0]
        #print("LEN:",length)
        # ---- HEARTBEAT: silenzio ----
        if length == 0:
            count_silent_frames  = count_silent_frames + 1
            
            # print("END:",now, last_voice_time, now - last_voice_time)
            # se solo heartbeat per troppo tempo → fine segmento
            if count_silent_frames < FRAME_TIMEOUT:
                #print(count_silent_frames, FRAME_TIMEOUT)
                continue
            else:
                win, buffer = build_window_from_buffer(buffer, final_flush=True)
                if win is None:
                    continue
        
                # ---- FEATURE EXTRACTION ----
                wav = torch.from_numpy(win).unsqueeze(0)  # (1, T)
                if wav.size(1) > int(SAMPLE_RATE / 10):  #remain wav length greater than 100ms
                    #print("\nCOUNT_SILENCE:", count_silent_frames, wav.size(1))
                    spec = spec_transform(wav, args)
                    spec = melspec_transform(spec, args).to(dev)

                    # 3) encoder sul chunk
                    valid_len = torch.tensor([spec.size(2)])
                    encoder = model(spec, valid_len)
                    enc = encoder[5]   # (B, T_enc, D)
                    enc_central = enc #[:, LB_e : LB_e + CK_e, :]
            
                    # 5) decodifica
                    transc = inf.stream_decoder(emission=enc_central, partial=True)
                    #print(" ".join(transc), end='\r')
                    #transc=" ".join(transc)
                    print_live_caption(transc)                
                
                count_silent_frames = 0
                final_flush = False
                continue

        # ---- CHUNK VOCALE ----
        data = recv_exact(conn, length)
        if data is None:
            print("END CONNECT!")
            break

        # int16 → float32 [-1, 1]

        pcm_buffer = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        #pcm = np.clip(pcm, -1.0, 1.0)
        
        buffer = np.concatenate([buffer, pcm_buffer])

        #print("LEN_1:",len(buffer))
        win, buffer = build_window_from_buffer(buffer, final_flush=False)
        #print("LEN_2:",len(buffer))
        
        if win is None:
            continue
        
        #print(f"[receiver] Finestra pronta: {len(win)} campioni")
        # --- Estrai SOLO la parte centrale (2 secondi) ---
        central = win[LB : LB + CK]   # float32 [-1,1]
        # --- Converti in int16 per salvataggio PCM ---
        pcm16 = (central * 32767).astype(np.int16)
        # --- Append al file PCM ---
        #central_f.write(pcm16.tobytes())
        #print(f"[receiver] Salvati {len(central)} campioni centrali")
        
        
        # ---- FEATURE EXTRACTION ----
        # win shape: (T,)
        wav = torch.from_numpy(win).unsqueeze(0)  # (1, T)
        #print("BUFF:",len(buffer), len(win), len(win[LB:LB+CK]))
        #print("WAV:",wav.size())
        
        #print("T2:", time.time())        
        # 1) feature extraction
        spec = spec_transform(wav, args)
        spec = melspec_transform(spec, args).to(dev)

        # 3) encoder sul chunk
        valid_len = torch.tensor([spec.size(2)])
        encoder = model(spec, valid_len)
        enc = encoder[5]   # (B, T_enc, D)
        B, T_full, D = enc.shape
        enc_central = enc[:, LB_e : LB_e + CK_e, :]

        # 5) decodifica
        transc = inf.stream_decoder(emission=enc_central, partial=True)
        #print("TRANSC:",transc)
        #print("Parziale: "," ".join(transc), end='\r')
        #print(" ".join(transc), end='\r')
        #transc=" ".join(transc)
        #
        print_live_caption(transc)                

    # flush finale se la connessione termina        
    final=inf.stream_decoder(partial=False)
    final_text = normalize_final_output(final)
    if final_text.strip():
        print("\nFINALE:", final_text)

    conn.close()
    sock.close()

    

def handler_batch(args, model, valid_len,  inf, dev, file: str):
    #audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    #waveform, sample_rate = torchaudio.load("/home/daniele/early-exit-transformer/2961-960-0000.flac")
    waveform, sample_rate = torchaudio.load(file)
    #waveform, sample_rate = torchaudio.load("/home/daniele/early-exit-transformer/test_stream.wav")


    spec = spec_transform(waveform, args)  # .to(device)
    spec = melspec_transform(spec, args).to(dev)
    valid_len = torch.tensor([spec.size(2)])
    encoder = model(spec.to(args.device), valid_len)

    transc = None
    
    if dev == "cpu":
        transc = inf.ctc_predict_(encoder[5])
        #transc = inf.stream_decoder(encoder[5])
        #print("Parziale:", " ".join(transc), end="\r")

        #print("Finale:", self.s_decoder.finalize())
        
    if dev == "cuda":        
        best_combined = inf.ctc_cuda_predict(encoder[5], args.tokens)
        transc = args.sp.decode(best_combined[0][0].tokens).lower()

    return normalize_output(transc)

    
    
def run(args, model, inf):

    valid_len = 0
    dev=args.device  #cuda #cpu
    if args.online_batch == False:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((HOST, PORT))
        sock.listen(1)
        print("WSL in ascolto...")
        conn, addr = sock.accept()
        print("Connesso a:", addr)
        handler_online(args, model, valid_len, inf, dev, conn)        
    else:
        transc = handler_batch(args, model, valid_len, inf, dev, file=args.file_path)
        print(f'{transc=}')

    return


def main():
    #
    #   CONFIG
    #

    args = get_args()

    lang = "eng"
    
    # If model checkpoint path is provided, load it.
    # (Overrides conf parameters)

    '''
    if lang == "eng":
        args.load_model_dir = os.getcwd() + '/English-EE-conformer'
        args.load_model_path = args.load_model_dir + "/english-EE-conformer"

    if lang == "it":
        args.load_model_dir = os.getcwd() + '/Italian-EE-conformer'
        args.load_model_path = args.load_model_dir + "/italian-EE-conformer"
    '''
    
    args.batch_size=1
    args.device="cpu"
    
    # Parse config from command line arguments

    # Define model
    print(args)

    model = Early_conformer(src_pad_idx=args.src_pad_idx,
                            n_enc_exits=args.n_enc_exits,
                            d_model=args.d_model,
                            enc_voc_size=args.enc_voc_size,
                            dec_voc_size=args.dec_voc_size,
                            max_len=args.max_len,
                            d_feed_forward=args.d_feed_forward,
                            n_head=args.n_heads,
                            n_enc_layers=args.n_enc_layers_per_exit,
                            features_length=args.n_mels,
                            drop_prob=args.drop_prob,
                            depthwise_kernel_size=args.depthwise_kernel_size,
                            device=args.device).to(args.device)


    model_path=args.load_model_dir+"/model"
    model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
    print(f'The model has {count_parameters(model):,} trainable parameters')
    #torch.multiprocessing.set_start_method('spawn')
    #torch.set_num_threads(args.n_threads)
    
    # Used to access various inference functions, see util/beam_infer
    inf = BeamInference(args=args)
    run(model=model, args=args, inf=inf)


if __name__ == '__main__':
    main()
