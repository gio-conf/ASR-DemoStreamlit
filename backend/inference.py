import argparse
import io
import os
import re
import socket
import struct
import sys
import time
from collections import deque

import numpy as np
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch import nn, optim
from torchaudio.models.decoder import ctc_decoder

from data import get_infer_data_loader
from models.model.early_exit import (
    Early_conformer,
    Early_zipformer,
    Splitformer,
    full_conformer,
)
from util.beam_infer import BeamInference
from util.conf import get_args
from util.data_loader import text_transform
from util.epoch_timer import epoch_time
from util.model_utils import *
from util.tokenizer import *


def spec_transform(waveform, args):
    spec_t = T.Spectrogram(
        n_fft=args.n_fft * 2, hop_length=args.hop_length, win_length=args.win_length
    )
    return spec_t(waveform)


def melspec_transform(waveform, args):
    melspec_t = T.MelScale(
        sample_rate=args.sample_rate, n_mels=args.n_mels, n_stft=args.n_fft + 1
    )
    return melspec_t(waveform)


#############################################
# PARAMETRI AUDIO E FINESTRE
#############################################

SAMPLE_RATE = 16000


LOOKBEHIND_SEC = 0.5  # 1.0 #1.0
CHUNK_SEC = 2.0
LOOKAHEAD_SEC = 0.5  # 1.0 #1.0

LB = int(LOOKBEHIND_SEC * SAMPLE_RATE)
CK = int(CHUNK_SEC * SAMPLE_RATE)
LA = int(LOOKAHEAD_SEC * SAMPLE_RATE)

WINDOW = LB + CK + LA  # 4 secondi = 64000 campioni
ADVANCE = CK  # avanza di 2 secondi = 32000 campioni

# buffer PCM per accumulare voce
# pcm_buffer = deque(maxlen=SAMPLE_RATE * 60)  # 60 s max

# per segmentazione basata su heartbeat
SEGMENT_TIMEOUT = 2000  # 800 ms di soli heartbeat = fine segmento
FRAME_MS = 30
FRAME_TIMEOUT = SEGMENT_TIMEOUT / FRAME_MS

#############################################
# MODELLO E DECODER (DA INTEGRARE)
#############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device=}")

# Esempio: da sostituire con il tuo codice
# model = load_model(args).to(device).eval()
# decoder = StreamingCTCBeamSearchLexicon(...)

#############################################
# SOCKET
#############################################


_last_len = 0


def print_live_caption(partial):
    global _last_len

    # Normalizza l’output
    if isinstance(partial, list):
        text = " ".join(partial)  # <-- qui mettiamo gli spazi
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


#############################################
# FUNZIONE: COSTRUISCI FINESTRA PER IL MODELLO
#############################################

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

    if final_flush:
        if len(buffer) == 0:
            return None, buffer
        window = buffer.copy()
        new_buffer = buffer[:0]  # svuota
        return window, new_buffer

    LB_SAMPLES = LB
    CK_SAMPLES = CK
    LA_SAMPLES = LA

    total_needed = LB_SAMPLES + CK_SAMPLES + LA_SAMPLES

    # Se non abbiamo abbastanza per una finestra completa
    if len(buffer) < CK_SAMPLES:
        # troppo poco per emettere qualcosa
        if not final_flush:
            return None, buffer
        # final flush: emetti tutto quello che c'e`
        window = buffer.copy()
        buffer = buffer[len(buffer) :]  # svuota
        return window, buffer

    # Se abbiamo abbastanza per il chunk centrale
    # ma non abbastanza per LB o LA  usiamo quello che c’e`
    if len(buffer) < total_needed:
        if not final_flush:
            # aspettiamo ancora look-ahead
            return None, buffer

        # final flush: emetti finestra elastica
        # prendiamo tutto cio` che c’e`
        window = buffer.copy()
        buffer = buffer[len(buffer) :]
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
LB_e = int(LOOKBEHIND_SEC * EMB_PS)
CK_e = int(CHUNK_SEC * EMB_PS)


# buffer = np.zeros(0, dtype=np.float32)
speech_detected = False
count_silent_frames = 0


def handler(args, model, valid_len, inf, dev, data: bytes, buffer, final: bool):
    # CPU MODE ONLY
    global speech_detected
    global count_silent_frames

    pcm_buffer = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    buffer = np.concatenate([buffer, pcm_buffer])

    final_text = ""

    if final:
        print("Final flush")
        win, buffer = build_window_from_buffer(buffer, final_flush=True)
        if win is None:
            transc = inf.stream_decoder(partial=False)
            print(f"{transc=}")
            return normalize_output(transc), np.zeros(0, dtype=np.float32)
        wav = torch.from_numpy(win).unsqueeze(0)  # (1, T)
        if wav.size(1) > int(SAMPLE_RATE / 10):  # remain wav length greater than 100ms
            # print("\nCOUNT_SILENCE:", count_silent_frames, wav.size(1))
            spec = spec_transform(wav, args)
            spec = melspec_transform(spec, args).to(dev)

            # 3) encoder sul chunk
            valid_len = torch.tensor([spec.size(2)])
            encoder = model(spec, valid_len)
            enc = encoder[5]  # (B, T_enc, D)

            enc_central = enc[:, LB_e : LB_e + CK_e, :]

            final_text += normalize_output(
                inf.stream_decoder(emission=enc_central, partial=True)
            )

        inf.stream_decoder(partial=False)
        return final_text, np.zeros(0, dtype=np.float32)

    while len(buffer) >= (LB + CK + LA):
        win, buffer = build_window_from_buffer(buffer, final_flush=False)

        if win is None:
            continue

        speech_detected = True

        wav = torch.from_numpy(win).unsqueeze(0)

        spec = spec_transform(wav, args)
        spec = melspec_transform(spec, args).to(dev)

        valid_len = torch.tensor([spec.size(2)])

        encoder = model(spec, valid_len)
        enc = encoder[5]  # (B, T_enc, D)

        enc_central = enc[:, LB_e : LB_e + CK_e, :]
        # 5) decodifica
        final_text += normalize_output(
            inf.stream_decoder(emission=enc_central, partial=True)
        )

    return final_text, buffer

    """
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

    return transc
    """


def run(args, model, inf):
    valid_len = 0
    dev = args.device  # cuda #cpu
    transc = handler(args, model, valid_len, inf, dev, data=bytes(0))
    print(f"{transc=}")

    return transc


class StreamingASR:
    def __init__(self):
        self.args = get_args()

        self.args.batch_size = 1
        self.args.device = "cpu"

        self.model = Early_conformer(
            src_pad_idx=self.args.src_pad_idx,
            n_enc_exits=self.args.n_enc_exits,
            d_model=self.args.d_model,
            enc_voc_size=self.args.enc_voc_size,
            dec_voc_size=self.args.dec_voc_size,
            max_len=self.args.max_len,
            d_feed_forward=self.args.d_feed_forward,
            n_head=self.args.n_heads,
            n_enc_layers=self.args.n_enc_layers_per_exit,
            features_length=self.args.n_mels,
            drop_prob=self.args.drop_prob,
            depthwise_kernel_size=self.args.depthwise_kernel_size,
            device=self.args.device,
        ).to(self.args.device)

        model_path = self.args.load_model_dir + "/model"

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.args.device, weights_only=True)
        )

        self.model.eval()

        self.inf = BeamInference(args=self.args)

    def transcribe(self, pcm_bytes):
        return handler_online(
            args=self.args,
            model=self.model,
            valid_len=0,
            inf=self.inf,
            dev=self.args.device,
            data=pcm_bytes,
        )


def main():
    #
    #   CONFIG
    #

    args = get_args()

    lang = "eng"

    # If model checkpoint path is provided, load it.
    # (Overrides conf parameters)

    """
    if lang == "eng":
        args.load_model_dir = os.getcwd() + '/English-EE-conformer'
        args.load_model_path = args.load_model_dir + "/english-EE-conformer"

    if lang == "it":
        args.load_model_dir = os.getcwd() + '/Italian-EE-conformer'
        args.load_model_path = args.load_model_dir + "/italian-EE-conformer"
    """

    args.batch_size = 1
    args.device = "cpu"

    # Parse config from command line arguments

    # Define model
    print(args)

    model = Early_conformer(
        src_pad_idx=args.src_pad_idx,
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
        device=args.device,
    ).to(args.device)

    model_path = args.load_model_dir + "/model"
    model.load_state_dict(
        torch.load(model_path, map_location=args.device, weights_only=True)
    )
    print(f"The model has {count_parameters(model):,} trainable parameters")
    # torch.multiprocessing.set_start_method('spawn')
    # torch.set_num_threads(args.n_threads)

    # Used to access various inference functions, see util/beam_infer
    inf = BeamInference(args=args)
    text = run(model=model, args=args, inf=inf)
    return text


if __name__ == "__main__":
    main()
