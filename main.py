import streamlit as st

####################################
# Variables
####################################
if "ctr" not in st.session_state:
    st.session_state["ctr"] = 0
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

def save_file(file):
    if file is not None:
        bytes_data = file.getvalue()
        with open(f"{AUDIO_DIR}/user_audio_{st.session_state.ctr}.mp3", "wb") as f:
            f.write(bytes_data)
        st.session_state.ctr += 1

# UI
st.title(""":blue[Transcriptor]""")

with st.container(
    border=True, height="stretch", width="stretch", horizontal_alignment="center"
):
    with st.container(horizontal_alignment="center"):
        chosen_mode = st.selectbox("""**:blue[Scegli la modalità]**""", ("File", "Microfono"), label_visibility="collapsed")
    
    st.divider()
    # with st.container(border=True, height="content", width="stretch"):
    # File upload
    if chosen_mode == "File":
        uploaded_file = st.file_uploader(
            "Upload file", type="audio", label_visibility="collapsed"
        )
        file_to_transcript = uploaded_file
        st.session_state.is_transcripted = False
    # Mic rec
    else:
        audio_batch = st.audio_input("Insert audio", label_visibility="collapsed")
        file_to_transcript = audio_batch

    st.divider()

    with st.container(horizontal=True, horizontal_alignment="distribute"):
        with st.popover("Settings"):
            # Scelta del linguaggio
            st.session_state.lang = st.selectbox("Seleziona lingua", ["Italian", "English"])

            # Scelta della task
            task_scelta = st.radio("Seleziona Task", ["ASR", "ST"])

            # Scelta del modello
            modello_scelto = st.selectbox(
                "Scegli il modello", options=["M1", "M2", "M3", "M4"]
            )

        # Si procede a trascrivere
        with st.container(horizontal_alignment="center", width="content"):
            if chosen_mode == "File":
                if st.button("Trascrivi", type="tertiary"):
                    if file_to_transcript is not None:
                        save_file(file_to_transcript)
                        # Chiamare logica di trascrizione
                        st.session_state.is_transcripted = True
                    else:
                        st.warning("Inserisci qualcosa da trascrivere")

    st.divider()

    if chosen_mode == "Microfono" and file_to_transcript is not None:
        st.html("<p>Qua apparira' la trascrizione</p>")
    elif chosen_mode == "File" and file_to_transcript is not None:
        st.write("Trascrizione")