from gtts import gTTS
from io import BytesIO
import base64
import streamlit as st

def autoplay_audio(file, autoplay):
    # b64 = base64.b64encode(file).decode()
    b64 = base64.b64encode(file.getvalue()).decode()
    if autoplay:
        md = f"""
            <audio id="audioTag" autoplay>
            <source src="data:audio/mp3;base64,{b64}"  type="audio/mpeg" format="audio/mpeg">
            </audio>
            """
    else:
        md = f"""
            <audio id="audioTag" >
            <source src="data:audio/mp3;base64,{b64}"  type="audio/mpeg" format="audio/mpeg">
            </audio>
            """
    st.markdown(
        md,
        unsafe_allow_html=True,
    )

def generate_tts_audio(text, lang='en'):
    sound_file = BytesIO()
    tts = gTTS(text, lang=lang)
    tts.write_to_fp(sound_file)
    return sound_file