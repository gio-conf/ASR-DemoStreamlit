import queue
import threading
import time

import av
import numpy as np
import requests
import streamlit as st
import webrtcvad
from pandas._libs.tslibs.nattype import iNaT
from scipy.signal import resample
from streamlit_webrtc import WebRtcMode, webrtc_streamer


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


# TO FIX, streamlit reruns the script everytime the user touches anything, so buf length is always 500ms
config = SharedConfig()

####################################
# Variables
####################################
if "ctr" not in st.session_state:
    st.session_state["ctr"] = 0
if "is_transcripted" not in st.session_state:
    st.session_state["is_transcripted"] = False
if "lang" not in st.session_state:
    st.session_state["lang"] = "it"
if "transcripted_text" not in st.session_state:
    st.session_state["transcripted_text"] = ""
if "exit" not in st.session_state:
    st.session_state["exit"] = "All"
if "rt_exit" not in st.session_state:
    st.session_state["rt_exit"] = (
        int(st.session_state.exit) - 1 if st.session_state.exit != "All" else 99
    )
if "realtime_content" not in st.session_state:
    st.session_state["realtime_content"] = [""] * 6
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = None
if "audio_started" not in st.session_state:
    st.session_state["audio_started"] = False
if "done" not in st.session_state:
    st.session_state["done"] = False
if "finished" not in st.session_state:
    st.session_state["finished"] = True

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


vad = webrtcvad.Vad(2)


def audio_frame_callback(frame: av.AudioFrame):
    audio = frame.to_ndarray()

    if frame.layout.name == "stereo":
        audio = audio.reshape(-1, 2)
        audio = audio.mean(axis=1)

    num_samples = audio.shape[0]
    target_num_samples = int(num_samples * 16000 / frame.sample_rate)

    audio = resample(audio, target_num_samples)
    audio = audio.astype(np.int16)

    if vad.is_speech(audio.tobytes(), 16000):
        audio_queue.put(audio)

    return frame


def on_audio_ended():
    get_audio_queue().put(SENTINEL)
    get_finish_queue().put(True)


def sender_worker(audio_queue):
    buf = bytearray()
    URL = "http://127.0.0.1:8000/chunks/"
    session_id = None
    f = None
    while True:
        try:
            data = audio_queue.get()

            target_ms = config.get_target_length()
            target_len = int(target_ms * 640 / 20)
            if isinstance(data, str) and data == SENTINEL:
                print("sentinel recv")
                if f is not None:
                    f.close()
                    f = None
                print("stopped sending")
                files = {
                    "file": (
                        "microfono.lastpart",
                        bytes(),
                        "application/octet-stream",
                    )
                }

                params = {}

                if session_id:
                    params["session_id"] = session_id
                    params["final"] = True
                    params["lang"] = "it"

                response = requests.post(
                    URL,
                    files=files,
                    data=params,
                )

                resp_json = response.json()
                session_id = resp_json["session_id"]
                if any(resp_json["result"]):
                    update_queue.put(resp_json)

                session_id = None
                continue

            audio_16 = data

            raw_data = audio_16.tobytes()

            buf.extend(raw_data)

            if len(buf) >= target_len:
                files = {
                    "file": (
                        "microfono.part0",
                        buf,
                        "application/octet-stream",
                    )
                }

                params = {}

                if session_id:
                    params["session_id"] = session_id

                params["lang"] = "it"
                params["final"] = False

                response = requests.post(
                    URL,
                    files=files,
                    data=params,
                )

                resp_json = response.json()
                session_id = resp_json["session_id"]
                if any(resp_json["result"]):
                    update_queue.put(resp_json)

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
                # WIP
                # new_length = st.number_input(
                #     "Target buffer length in ms",
                #     min_value=200,
                #     max_value=1000,
                #     value=config.get_target_length(),
                #     step=20,
                # )

                # config.set_target_length(new_length)
                # Scelta del linguaggio
                st.session_state.lang = st.selectbox(
                    "Seleziona lingua",
                    key="mic_rt_chosen_lang",
                    options=["it", "en"],
                )

                st.session_state.exit = st.selectbox(
                    "Scegli l'uscita",
                    key="mic_rt_chosen_exit",
                    options=["All", "1", "2", "3", "4", "5", "6"],
                )

            if ctx.state.playing and not st.session_state["audio_started"]:
                st.session_state.realtime_content = [""] * 6
                st.session_state.rt_exit = (
                    int(st.session_state.exit) - 1
                    if st.session_state.exit != "All"
                    else 99
                )
                st.session_state["audio_started"] = True
                requests.post(
                    "http://127.0.0.1:8000/set_exit/",
                    data={"new_exit": st.session_state.rt_exit},
                )

            st.divider()

            history_boxes = [st.empty() for _ in range(6)]

            transcript = ""

            poll_interval = 0.2
            while ctx.state.playing:
                try:
                    resp_json = update_queue.get_nowait()
                    result = resp_json.get("result", [])
                    for r in result:
                        text = r["text"]
                        exit = r["exit"]
                        st.session_state["realtime_content"][exit] += text + " "
                except queue.Empty:
                    pass

                if st.session_state.rt_exit == 99:
                    for i, hb in enumerate(history_boxes):
                        hb.write(
                            f"Exit {i + 1}: {st.session_state['realtime_content'][i]}"
                        )
                else:
                    history_boxes[0].write(
                        f"Exit {st.session_state.rt_exit + 1}: {st.session_state['realtime_content'][st.session_state.rt_exit]}"
                    )

                time.sleep(poll_interval)

            st.divider()

            final_boxes = [st.empty() for _ in range(6)]

            try:
                st.session_state["finished"] = finish_queue.get_nowait()
            except queue.Empty:
                st.session_state["finished"] = True

            # After stopping
            if st.session_state["finished"]:
                try:
                    st.session_state["audio_started"] = False
                    resp_json = update_queue.get_nowait()
                    result = resp_json.get("result", [])
                    if st.session_state.rt_exit == 99:
                        for r in result:
                            text = r["text"]
                            exit = r["exit"]
                            st.session_state["realtime_content"][exit] += text

                            for i in range(len(st.session_state.realtime_content)):
                                st.session_state.realtime_content[i] += text + " "
                                final_boxes[i].write(
                                    f"Exit {i + 1}: {st.session_state.realtime_content[i]}"
                                )
                    else:
                        st.session_state.realtime_content[st.session_state.rt_exit] += (
                            result[0]["text"] + " "
                        )
                        final_boxes[0].write(
                            f"Exit {st.session_state.rt_exit + 1}: {st.session_state.realtime_content[st.session_state.rt_exit]}"
                        )
                except queue.Empty:
                    pass

            st.session_state.rt_exit = (
                int(st.session_state.exit) - 1 if st.session_state.exit != "All" else 99
            )

            if st.session_state.rt_exit == 99:
                for i in range(len(st.session_state.realtime_content)):
                    final_boxes[i].write(
                        f"Exit {i + 1}: {st.session_state.realtime_content[i]}"
                    )
            else:
                final_boxes[0].write(
                    f"Exit {int(st.session_state.rt_exit) + 1}: {st.session_state.realtime_content[st.session_state.rt_exit]}"
                )
        else:
            st.warning("Load model from sidebar")


def file_tab_fn(mic_mode=False, key=""):

    with st.container(
        border=True, height="stretch", width="stretch", horizontal_alignment="center"
    ):
        if True:  # st.session_state["model_loaded"]:
            if mic_mode:
                file = st.audio_input(
                    "Audio", label_visibility="collapsed", sample_rate=16000
                )
            else:
                file = st.file_uploader(
                    "Upload file",
                    type="audio",
                    label_visibility="collapsed",
                )

            st.session_state.is_transcripted = False

            st.divider()

            with st.container(horizontal=True, horizontal_alignment="distribute"):
                with st.popover("Settings"):
                    # language selection
                    st.session_state.lang = st.selectbox(
                        "Select language",
                        key=f"{key}_file_chosen_lang",
                        options=["it", "en"],
                    )

                    # Exit selection
                    exit = st.selectbox(
                        "Select exit",
                        key=f"{key}_file_chosen_exit",
                        options=["All", "1", "2", "3", "4", "5", "6"],
                    )
                    st.session_state.exit = int(exit) - 1 if exit != "All" else 99

                with st.container(horizontal_alignment="center", width="content"):
                    if st.button(
                        "Transcribe", key=f"{key}_file_transcribe_btn", type="tertiary"
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
                            params = {
                                "lang": st.session_state.lang,
                                "exit": st.session_state.exit,
                            }
                            resp = requests.post(
                                "http://127.0.0.1:8000/uploads/",
                                files=files,
                                data=params,
                            )
                            st.session_state["transcripted_text"] = resp.json()[
                                "result"
                            ]
                            st.session_state.is_transcripted = True
                        else:
                            st.error("Nulla da trascrivere")

            st.divider()

            transc = st.session_state["transcripted_text"]
            for t in transc:
                st.write(f"Exit {t['exit'] + 1}: {t['text']}")
        else:
            st.warning("Load model from sidebar")


with file_tab:
    file_tab_fn(key="file")

with mic_file_tab:
    file_tab_fn(mic_mode=True, key="mic")


# Credits
# with st.container(horizontal_alignment="center"):
#     st.html("<p>Made by Giovanni Confente Broll Avila</p>")
