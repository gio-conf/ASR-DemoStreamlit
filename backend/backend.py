from fastapi import FastAPI, File, UploadFile, Form, WebSocket
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Annotated
import shutil
from inference_online import handler_batch
from inference import handler
from torch import nn, optim
import torchaudio
from torchaudio.models.decoder import ctc_decoder
from data import get_infer_data_loader
from models.model.early_exit import Early_conformer
from util.beam_infer import BeamInference
from util.conf import get_args
from util.data_loader import text_transform
from util.epoch_timer import epoch_time
from util.model_utils import *
from util.tokenizer import *
import torchaudio.transforms as T
import torch.nn.functional as F
import numpy as np

def load_audio(audio_bytes, target_sr=16000):

    waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))

    # stereo -> mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # float32 [-1,1] -> int16
    audio_int16 = (waveform.squeeze(0).numpy() * 32767.0).astype(np.int16)

    # ritorna buffer bytes
    return audio_int16.tobytes()

UPLOAD_DIR = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_loaded
    global UPLOAD_DIR
    global it_model, it_args, it_inf, it_valid_len, it_dev
    global en_model, en_args, en_inf, en_valid_len, en_dev
    print("App is starting...")
    UPLOAD_DIR = Path("uploads")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    it_args = get_args([], "Italian")
    it_model, it_inf, it_valid_len, it_dev = await load_model(it_args)
    print("IT-model loaded")

    en_args = get_args([], "English")
    en_model, en_inf, en_valid_len, en_dev = await load_model(en_args)
    print("EN-model loaded")
    model_loaded = True
    yield
    print("App is shutting down...")
    if UPLOAD_DIR.exists():
        for item in UPLOAD_DIR.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception:
                pass

app = FastAPI(lifespan=lifespan)
model_loaded: bool = False
it_model = None
it_args = []
it_inf = None
it_valid_len = None
it_dev = None
en_model = None
en_args = []
en_inf = None
en_valid_len = None
en_dev = None

@app.get("/model_info")
def get_model_info():
    return {"state": model_loaded}

@app.post("/uploads/")
async def upload(file: UploadFile = File(...), lang: str = Form("Italian")):
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as out_file:
        while content := await file.read(1024*1024):
            out_file.write(content)
    await file.close()
    if lang == "Italian":
        print("Italian transcription")
        transc = handler_batch(it_args, it_model, it_valid_len, it_inf, it_dev, f"uploads/{file.filename}")
    else:
        print("English transcription")
        transc = handler_batch(en_args, en_model, en_valid_len, en_inf, en_dev, f"uploads/{file.filename}")
    return {'text': transc}

@app.post("/set_model/")
async def set_model(lang: str = Form("Italian")):
    global args
    args = get_args([], lang.capitalize())
    await load_model(args)
    return {'lang': lang}

class Session:
    def __init__(self):
        self.buffer = np.zeros(0, dtype=np.float32)

sessions = {}
session_cnt = 0

@app.post("/chunks/")
async def handle_chunk(file: Annotated[bytes, File()], session_id: str | None = Form(None)): 
    global session_cnt
    
    s = sessions.get(session_id)
    print(f'{session_id=}')
    if s is None:
        s = Session()
        session_cnt += 1
        session_id = str(session_cnt)
        sessions[session_id] = s

    transc, s.buffer = handler(args, model, valid_len, inf, dev, data=file, buffer=s.buffer)

    if transc != "":
        print(f'{transc=}')

    return {"text": transc, 'session_id': session_id}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global session_cnt
    
    # Da rivedere
    s = sessions.get(1)
    if s is None:
        s = Session()
        session_cnt += 1
        session_id = str(session_cnt)
        sessions[session_id] = s
    await websocket.accept()
    while True:
        data = await websocket.receive_bytes()
        # keep handler unchanged; await its result and unpack
        transc, s.buffer = handler(args, model, valid_len, inf, dev, data=data, buffer=s.buffer, final=False)
        await websocket.send_text(transc)

async def load_model(args):
    global inf, valid_len, dev
    # If model checkpoint path is provided, load it.
    # (Overrides conf parameters)

    """
    if lang == "eng":
        args.load_model_dir = os.getcwd() + '/English-EE-conformer'
        args.load_model_path = args.load_model_dir + "/english-EE-conformer"

    if lang == "it":
        args.load_model_dir = os.getcwd() + '/Italian-EE-conformer'
        args.load_model_path = args.load_model_dir + "/italian-EE-conformer"    """

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

    inf = BeamInference(args=args)
    valid_len = 0
    dev = args.device

    return model, inf, valid_len, dev


# @app.post("/files/")
# async def handle_file(file: Annotated[bytes, File()]):
#     data = load_audio(file)
#     transc = handler(args, model, valid_len, inf, dev, data=data)

#     return {"text": transc}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)
