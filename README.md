# ASR_DemoStreamlit

This repository contains a full web application used to transcribe user audio.
There are 3 main features:
- Batch recording and transcription
- File upload and transcription
- Real-time transcription (WIP)

Currently the only supported model is this one https://huggingface.co/SpeechTek/Italian-EE-conformer and its english counterpart.

## How to run
Advice: I developed this is a virtual enviroment with python 3.11.14.
- Clone the repository
- Inside of /backend, put both models' directories
- Install the the requirements with ```pip install -r requirements.txt``` in the root directory
- Open a second terminal, with the same virtual enviroment
- In the first terminal, get inside of the backend directory and launch the backend server with ```python backend.py```
- After both models are loaded and the server is ready. Launch the UI with ```streamlit run ./frontend/main.py```

## N.B.
There is an error with streamlit-webrtc. So before running, Modify the shutdown.py file in ```.venv/lib/python3.11/site-packages/streamlit_webrtc/``` line 126.
Change it from ```if self._polling_thread.is_alive():``` to ```if self._polling_thread is not None and self._polling_thread.is_alive():```.
