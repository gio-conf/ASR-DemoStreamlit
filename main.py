import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer

####################################
# Variables
####################################
if "ctr" not in st.session_state:
    st.session_state["ctr"] = 0
if "is_transcripted" not in st.session_state:
    st.session_state["is_transcripted"] = False
if "lang" not in st.session_state:
    st.session_state['lang'] = "Italian"
AUDIO_DIR = "./user_files"
file_to_transcript = ""

####################################
# CSS Styling
####################################

st.markdown("""
<style>
    h1 {
        font-size: 16px;
        text-align: center;
        text-transform: uppercase;
    }
    p {
        font-size: 24px;    
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

####################################
# Functions
####################################

def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame: 
    # Apply volume control to audio samples
    samples = frame.to_ndarray()
    samples = (volume * samples).astype(samples.dtype)  # type: ignore
    # Create new frame with processed audio
    new_frame = av.AudioFrame.from_ndarray(samples, layout=frame.layout.name)
    new_frame.sample_rate = frame.sample_rate
        
    return new_frame

def save_file(file):
    if file is not None:
        bytes_data = file.getvalue()
        with open(f"{AUDIO_DIR}/user_audio_{st.session_state.ctr}.mp3", "wb") as f:
            f.write(bytes_data)
        st.session_state.ctr += 1

# UI
st.title(""":blue[Transcriptor]""")
mic_tab, file_tab = st.tabs(["Microfono", "Upload file"])

with mic_tab:
    with st.container(
        border=True, height="stretch", width="stretch", horizontal_alignment="center"
    ):
        
        volume = st.slider("Volume", 0.0, 2.0, 1.0, 0.1)
        webrtc_streamer(    
            key="audio",
            audio_frame_callback=audio_frame_callback,
            media_stream_constraints={"video": False, "audio": True},
        )


        st.divider()

        with st.container(horizontal=True, horizontal_alignment="distribute"):
            with st.popover("Settings"):
                # Scelta del linguaggio
                st.session_state.lang = st.selectbox("Seleziona lingua", key="mic_chosen_lang", options=["Italian", "English"])

                # Scelta della task
                task_scelta = st.radio("Seleziona Task", key="mic_chosen_task", options=["ASR", "ST"])

                # Scelta dell'uscita
                uscita_scelta = st.selectbox(
                    "Scegli l'uscita", key="mic_chosen_exit", options=["1", "2", "3", "4", "5", "6"]
                )

        st.divider()
        st.write("Trascrizione...")

with file_tab:
    with st.container(
        border=True, height="stretch", width="stretch", horizontal_alignment="center"
        ):
        # File upload
        uploaded_file = st.file_uploader(
            "Upload file", type="audio", label_visibility="collapsed"
        )
        file_to_transcript = uploaded_file
        st.session_state.is_transcripted = False

        st.divider()

        with st.container(horizontal=True, horizontal_alignment="distribute"):
            with st.popover("Settings"):
                # Scelta del linguaggio
                st.session_state.lang = st.selectbox("Seleziona lingua", ["Italian", "English"])

                # Scelta della task
                task_scelta = st.radio("Seleziona Task", ["ASR", "ST"])

                # Scelta dell'uscita
                uscita_scelta = st.selectbox(
                    "Scegli l'uscita", options=["1", "2", "3", "4", "5", "6"]
                )

            # Si procede a trascrivere
            with st.container(horizontal_alignment="center", width="content"):
                if st.button("Trascrivi", type="tertiary"):
                    if file_to_transcript is not None:
                        save_file(file_to_transcript)
                        # Chiamare logica di trascrizione
                        st.session_state.is_transcripted = True

        st.divider()

        if file_to_transcript is not None and st.session_state.is_transcripted:
            st.write("Trascrizione")
        else:
            st.error("Nulla da trascrivere")