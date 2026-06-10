import datetime
import queue
import threading
import time

import av
import numpy as np
import requests
import streamlit as st
import webrtcvad
from scipy.signal import resample
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from websockets.sync.client import connect


class SharedConfig:
    def __init__(self):
        self._lock = threading.Lock()
        self._target_length_ms = 500

    def set_target_length(self, value):
        with self._lock:
            self._target_length_ms = value

    def get_target_length(self):
        with self._lock:
            return self._target_length_ms


config = SharedConfig()


####################################
# Variables
####################################
if "ctr" not in st.session_state:
    st.session_state["ctr"] = 0
if "is_transcripted" not in st.session_state:
    st.session_state["is_transcripted"] = False
if "lang" not in st.session_state:
    st.session_state["lang"] = "Italian"
if "is_transcripted" not in st.session_state:
    st.session_state["is_transcripted"] = False
if "transcripted_text" not in st.session_state:
    st.session_state["transcripted_text"] = ""
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = None
if "exit" not in st.session_state:
    st.session_state["exit"] = "All"
if "realtime_content" not in st.session_state:
    st.session_state["realtime_content"] = ""

AUDIO_DIR = "user_files/"
file_to_transcript = ""


SENTINEL = "STOP"


@st.cache_resource
def get_audio_queue():
    return queue.Queue()


@st.cache_resource
def get_update_queue():
    return queue.Queue()


@st.cache_resource
def get_finish_queue():
    return queue.Queue()


update_queue = get_update_queue()
audio_queue = get_audio_queue()
finish_queue = get_finish_queue()

if "finished" not in st.session_state:
    st.session_state["finished"] = False


def audio_frame_callback(frame: av.AudioFrame):
    audio = frame.to_ndarray()

    if frame.layout.name == "stereo":
        audio = audio.reshape(-1, 2)
        audio = audio.mean(axis=1)

    num_samples = audio.shape[0]
    target_num_samples = int(num_samples * 16000 / 48000)

    audio = resample(audio, target_num_samples)
    audio = audio.astype(np.int16)
    audio_queue.put(audio)

    return frame


def on_audio_ended():
    get_audio_queue().put(SENTINEL)
    get_finish_queue().put(True)


def sender_worker(audio_queue):
    buf = bytearray()
    vad = webrtcvad.Vad(2)

    # f = open("out.raw", 'wb')
    with connect("ws://127.0.0.1:8000/ws") as websocket:
        while True:
            try:
                data = audio_queue.get()

                target_ms = config.get_target_length()
                target_len = int((target_ms / 1000) * 16000)
                if isinstance(data, str) and data == SENTINEL:
                    websocket.send(data)
                    msg = websocket.recv()
                    update_queue.put_nowait(msg)
                    print("sentinel recv")
                    print("stopped sending")
                    # f.flush()
                    # f.close()
                    # f = None
                    continue
                else:
                    # Debug audio
                    # if f is None:
                    #     f = open("out.raw", 'wb')

                    audio_16 = data

                    is_speech = vad.is_speech(audio_16, 16000)

                    raw_data = audio_16.tobytes()

                    buf.extend(raw_data)

                    if is_speech and len(buf) >= target_len:
                        print(f"{len(buf)=}")
                        # f.write(buf)
                        print(f"{datetime.datetime.now()=}")
                        websocket.send(buf)
                        message = websocket.recv()
                        update_queue.put_nowait(message)
                        buf = bytearray()
            except queue.Empty:
                continue


if "worker_thread" not in st.session_state:
    print("starting new thread")
    t = threading.Thread(target=sender_worker, args=(audio_queue,), daemon=True)
    t.start()
    st.session_state.worker_thread = t

####################################
# CSS Styling
####################################

st.markdown(
    """
<style>
    h1 {
        font-size: 24px;
        text-align: center;
        text-transform: uppercase;
    }
</style>
""",
    unsafe_allow_html=True,
)

####################################
# Functions
####################################

# UI
st.title("Transcriber")

# Sidebar
if not st.session_state["model_loaded"]:
    with st.sidebar:
        # Se il server dovesse essere partito ma i modelli non sono caricati
        resp = requests.get("http://127.0.0.1:8000/model_info")
        st.session_state.model_loaded = resp.json()["state"]

        if st.session_state.model_loaded:
            st.success("OK")
        else:
            st.error("Error")

# TABS
mic_file_tab, file_tab, mic_rt_tab = st.tabs(
    ["Record & Transcribe", "Upload & Transcribe", "Real-Time transcription"]
)

with mic_rt_tab:
    with st.container(
        border=True, height="stretch", width="stretch", horizontal_alignment="center"
    ):
        if st.session_state["model_loaded"]:
            ctx = webrtc_streamer(
                key="audio",
                audio_frame_callback=audio_frame_callback,
                on_audio_ended=on_audio_ended,
                mode=WebRtcMode.SENDONLY,
                media_stream_constraints={"video": False, "audio": True},
            )

            st.divider()
            with st.container(horizontal=True, horizontal_alignment="distribute"):
                with st.popover("Settings"):
                    new_length = st.number_input(
                        "Target buffer length in ms",
                        min_value=200,
                        max_value=1000,
                        value=config.get_target_length(),
                        step=40,
                    )

                    config.set_target_length(new_length)

                    st.session_state.lang = st.selectbox(
                        "Seleziona lingua",
                        key="mic_rt_chosen_lang",
                        options=["it", "en"],
                    )

                    st.write("WIP")
                    # Scelta della task
                    # task_scelta = st.radio("Seleziona Task", key="mic_rt_chosen_task", options=["ASR"])

                    # Scelta dell'uscita
                    # st.session_state.exit = st.selectbox(
                    #     "Scegli l'uscita", key="mic_rt_chosen_exit", options=["All", "1", "2", "3", "4", "5", "6"]
                    # )

            if ctx.state.playing and not st.session_state["audio_started"]:
                st.session_state["audio_started"] = True
                resp = requests.post(
                    "http://127.0.0.1:8000/model_specs/",
                    data={"lang": st.session_state.lang},
                )

            st.divider()
            history_box = st.empty()

            transcript = ""

            poll_interval = 0.2
            while ctx.state.playing:
                try:
                    while True:
                        resp_json = update_queue.get_nowait()
                        text = resp_json
                        if text:
                            transcript = text
                            history_box.write(transcript)
                            st.session_state["realtime_content"] = transcript
                except queue.Empty:
                    pass

                time.sleep(poll_interval)

            st.divider()

            try:
                st.session_state["finished"] = finish_queue.get_nowait()
            except queue.Empty:
                pass

            if st.session_state["finished"]:
                try:
                    st.write(st.session_state["realtime_content"])
                except queue.Empty:
                    pass

        else:
            st.warning("Carica il modello dalla sidebar")

with file_tab:
    with st.container(
        border=True, height="stretch", width="stretch", horizontal_alignment="center"
    ):
        if st.session_state["model_loaded"]:
            # File upload
            uploaded_file = st.file_uploader(
                "Upload file",
                type=[".wav", ".ogg", ".flac"],
                label_visibility="collapsed",
            )

            st.session_state.is_transcripted = False

            st.divider()

            with st.container(horizontal=True, horizontal_alignment="distribute"):
                with st.popover("Settings"):
                    # Scelta del linguaggio
                    st.session_state.lang = st.selectbox(
                        "Select language", ["Italian", "English"]
                    )

                    # Scelta della task
                    task_scelta = st.radio("Select Task", ["ASR"])

                    # Scelta dell'uscita
                    st.session_state.exit = st.selectbox(
                        "Select exit", options=["All", "1", "2", "3", "4", "5", "6"]
                    )

                # Si procede a trascrivere
                with st.container(horizontal_alignment="center", width="content"):
                    if st.button("Transcribe", type="tertiary"):
                        file_to_transcript = uploaded_file
                        file_to_transcript.name = "user_uploaded_file.wav"
                        if file_to_transcript is not None:
                            files = {
                                "file": (
                                    file_to_transcript.name,
                                    file_to_transcript,
                                    "audio/wav",
                                )
                            }
                            try:
                                resp = requests.post(
                                    "http://127.0.0.1:8000/uploads", files=files
                                )
                                st.session_state["transcripted_text"] = resp.json()[
                                    "text"
                                ]
                                st.session_state.is_transcripted = True
                            except ConnectionError:
                                st.error("Server irraggiungibile")
                        else:
                            st.error("Nulla da trascrivere")

            st.divider()

            if file_to_transcript is not None and st.session_state.is_transcripted:
                if st.session_state["exit"] == "All":
                    for i in range(len(st.session_state["transcripted_text"])):
                        st.write(
                            f"Exit {i + 1}: " + st.session_state["transcripted_text"][i]
                        )
                else:
                    chosen_idx = int(st.session_state["exit"]) - 1
                    st.write(
                        f"Exit {chosen_idx + 1}: "
                        + st.session_state["transcripted_text"][chosen_idx]
                    )
        else:
            st.warning("Carica il modello dalla sidebar")

with mic_file_tab:
    with st.container(
        border=True, height="stretch", width="stretch", horizontal_alignment="center"
    ):
        if st.session_state["is_model_loaded"]:
            file = st.audio_input(
                "Audio", label_visibility="collapsed", sample_rate=16000
            )
            st.session_state.is_transcripted = False

            st.divider()

            with st.container(horizontal=True, horizontal_alignment="distribute"):
                with st.popover("Settings"):
                    # Scelta del linguaggio
                    st.session_state.lang = st.selectbox(
                        "Select language",
                        key="mic_file_chosen_lang",
                        options=["Italian", "English"],
                    )

                    # Scelta della task
                    task_scelta = st.radio(
                        "Select Task", key="mic_lang_chosen_task", options=["ASR"]
                    )

                    # Scelta dell'uscita
                    st.session_state.exit = st.selectbox(
                        "Select exit",
                        key="mic_file_chosen_exit",
                        options=["All", "1", "2", "3", "4", "5", "6"],
                    )

                # Si procede a trascrivere
                with st.container(horizontal_alignment="center", width="content"):
                    if st.button(
                        "Transcribe", key="mic_file_transcribe_btn", type="tertiary"
                    ):
                        file_to_transcript = file
                        if file_to_transcript is not None:
                            file_to_transcript.name = "user_file.wav"
                            files = {
                                "file": (
                                    file_to_transcript.name,
                                    file_to_transcript,
                                    "audio/wav",
                                )
                            }
                            resp = requests.post(
                                "http://127.0.0.1:8000/uploads/", files=files
                            )
                            st.session_state["transcripted_text"] = resp.json()["text"]
                            st.session_state.is_transcripted = True
                        else:
                            st.error("Nulla da trascrivere")

            st.divider()

            if file_to_transcript is not None and st.session_state.is_transcripted:
                if st.session_state["exit"] == "All":
                    for i in range(len(st.session_state["transcripted_text"])):
                        st.write(
                            f"Exit {i + 1}: " + st.session_state["transcripted_text"][i]
                        )
                else:
                    chosen_idx = int(st.session_state["exit"]) - 1
                    st.write(
                        f"Exit {chosen_idx + 1}: "
                        + st.session_state["transcripted_text"][chosen_idx]
                    )
        else:
            st.warning("Carica il modello dalla sidebar")


# st.caption("Made by Giovanni Confente Broll Avila")
