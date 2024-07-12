import streamlit as st
from streamlit_chat import message
import tempfile
import os
from groq import Groq
from tts import autoplay_audio, generate_tts_audio
from streamlit_mic_recorder import mic_recorder

# message = """
# import streamlit as st

# st.write("hi")
# """

# with stylable_container(
#     "codeblock",
#     """
#                 code {
#                     white-space: pre-wrap !important;
#                     font-family: "Source Sans Pro", sans-serif !important;
#                     font-size: 1rem !important;
#                 }
#                 """,
# ):
#     st.code(message, language=None, line_numbers=False)

# st.markdown(message)

def transcribe_audio_groq(audio_file_path):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    with open(audio_file_path, "rb") as file:
        translation = client.audio.translations.create(
            file=(audio_file_path, file.read()),
            model="whisper-large-v3",
            prompt="Specify context or spelling",  # Optional
            response_format="json",  # Optional
            temperature=0.0  # Optional
        )
    return translation.text

 
from st_multimodal_chatinput import multimodal_chatinput

from carnet import recognize_file

def process_image(file_path):
    with open(file_path, "rb") as f:
        file_binary = f.read()

    result = recognize_file(file_path, file_binary)

    if 'error' in result:
        print("Error:", result['error'])
    else:
        # Extract data if no error
        car_info = result.get("car", {})
        color_info = result.get("color", {})
        angle_info = result.get("angle", {})
        bbox_info = result.get("bbox", {})

        print("Car Information:", car_info)
        print("Color Information:", color_info)
        print("Angle Information:", angle_info)
        print("Bounding Box Information:", bbox_info)


     
import base64
def base64_to_binary(base64_str):
    decoded_bytes = base64.b64decode(base64_str)

    binary_str = ''.join(format(byte, '08b') for byte in decoded_bytes)

    return binary_str

import re
from llm import invoke_model  # Import the function to invoke the model


def main():
    # st.set_page_config(page_title="ExxonMobil Customer Support Chatbot", page_icon="🛢️", layout="wide", initial_sidebar_state="collapsed")
    # st.set_page_config(page_title="ExxonMobil Customer Support Chatbot", page_icon="🛢️", layout="wide", initial_sidebar_state="collapsed")
    st.title("ExxonMobil Customer Support Chatbot")
    # print(get_past_and_generated())
    with st.sidebar.container():
        st.markdown(
            """
            <div style="font-size:0.75rem; color:gray;">
                ExxonMobil Chat can make mistakes. Check important info.
            </div>
            """, 
            unsafe_allow_html=True
        )
    def conversation_chat(query):
        result = invoke_model(query)
        st.session_state['history'].append((query, result['final_email']))
        return result['final_email']

    def initialize_session_state():
        # initialize_session_state1()
        if "history" not in st.session_state:
            st.session_state['history'] = []
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Ask me anything"]
        if "past" not in st.session_state:
            st.session_state['past'] = ['Hey!']
        if "my_recorder_output" not in st.session_state:
            st.session_state['my_recorder_output'] = None
    

    def display_chat_history():
        # prompt_input = st.chat_input("Enter the customer's question about ExxonMobil here:")
        # chatinput = multimodal_chatinput()

        col = st.columns([0.2, 0.2, 0.6, 0.2])
        # prompt_input = chatinput["text"] ##submitted text
        # uploaded_images = chatinput["images"] ##list of base64 encodings of uploaded images

        # Initialize the mic_recorder
        # with col[0]:
        #     mic_recorder(start_prompt="Speech To text", key='my_recorder', callback=process_audio)

        with col[0]:
            with st.container(border=True):
                mic_recorder(start_prompt="Speech To text",use_container_width=True, key='my_recorder',)# callback=process_audio)
                Audio = st.checkbox("Enable Audio", value=True)

        transcription_text = None
        if st.session_state.get('my_recorder_output'):
            audio_bytes = st.session_state.my_recorder_output['bytes']
            
            # Save audio to a file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
                temp.write(audio_bytes)
                temp_path = temp.name

            try:
                # Display path and try to transcribe audio
                # st.write(f"Temporary audio file saved at: {temp_path}")
                transcription_text = transcribe_audio_groq(temp_path)

            except Exception as e:
                st.error(f"Error in processing audio: {str(e)}")
            
            finally:
                os.remove(temp_path)

        st.write("Transcription:", transcription_text)


        from streamlit_extras.stylable_container import stylable_container
        with stylable_container(
            key="container_with_border",
            css_styles="""
                {
                    div[data-testid="stVerticalBlock"] {
                    position: fixed;
                    bottom: -8px;
                    width: 100%;
                    background-color: #0E117;
                    z-index: 1000; 
                }
                """,
        ):
            with st.container():
                user_inp = multimodal_chatinput()
        prompt_input = False
        images = False
        if user_inp:
            prompt_input = user_inp['text']
            images = user_inp['images']
            
        if prompt_input and images:
            base64_str = user_inp['images'][0].split(',')[1]  # Assuming the data is in the format "data:image/png;base64,<base64_encoded_data>"

            # Extract the file extension
            file_extension = re.search(r'(?<=image\/)[^;]+', user_inp['images'][0]).group(0)

            # Decode base64 to binary
            binary_data = base64.b64decode(base64_str)
            import random
            temp_file = f"serious_{random.randint(0, 10000)}.{file_extension}"
            result = recognize_file(f"{temp_file}", binary_data, file_extension)
            if 'error' in result:
                st.write("Error:", result['error'])
            # prompt = "What is my car version:"
            car_info = result["car"]
            string_result = f"""
            Car Information:
                Make: {car_info["make"]}
                Model: {car_info["model"]}
                Generation: {car_info["generation"]}
                Years: {car_info["years"]}
                Probability: {car_info["prob"]}%
                """
            car_info = string_result + prompt_input
            if car_info:
                inputs = {"initial_email":  car_info, "num_steps": 0}
                output = conversation_chat(inputs)
                st.session_state['past'].append( prompt_input)
                st.session_state['generated'].append(output)
                if Audio and output:
                    sound_file = generate_tts_audio(output)
                    autoplay_audio(sound_file, Audio)

        elif transcription_text is not None:
            inputs = {"initial_email": transcription_text, "num_steps": 0}
            output = conversation_chat(inputs)
            st.session_state['past'].append(transcription_text)
            st.session_state['generated'].append(output)
            if Audio and output:
                sound_file = generate_tts_audio(output)
                autoplay_audio(sound_file, Audio)
                transcription_text = None
            
        elif prompt_input:
            inputs = {"initial_email":  prompt_input, "num_steps": 0}
            output = conversation_chat(inputs)
            st.session_state['past'].append( prompt_input)
            st.session_state['generated'].append(output)
            if Audio and output:
                sound_file = generate_tts_audio(output)
                autoplay_audio(sound_file, Audio)
        elif images:
            base64_str = user_inp['images'][0].split(',')[1]  # Assuming the data is in the format "data:image/png;base64,<base64_encoded_data>"

            # Extract the file extension
            file_extension = re.search(r'(?<=image\/)[^;]+', user_inp['images'][0]).group(0)

            # Decode base64 to binary
            binary_data = base64.b64decode(base64_str)

            # Now binary_data contains the binary representation of the image
            # print("File extension:", file_extension)
            # st.write("File extension:", file_extension)
            # # print("Binary data:", binary_data)
            result = recognize_file("q.jpg", binary_data, file_extension)
            if 'error' in result:
                st.write("Error:", result['error'])
            # prompt = "What is my car version:"
            car_info = result["car"]
            color_info = result["color"]
            angle_info = result["angle"]
            bbox_info = result["bbox"]

            string_result = f"""
            Car Information:
                Make: {car_info["make"]}
                Model: {car_info["model"]}
                Generation: {car_info["generation"]}
                Years: {car_info["years"]}
                Probability: {car_info["prob"]}%

            Color Information:
                Name: {color_info["name"]}
                Probability: {color_info["probability"] * 100}%

            Angle Information:
                Name: {angle_info["name"]}
                Probability: {angle_info["probability"] * 100}%

            Bounding Box Information:
                Top-Left X: {bbox_info["tl_x"]}
                Top-Left Y: {bbox_info["tl_y"]}
                Bottom-Right X: {bbox_info["br_x"]}
                Bottom-Right Y: {bbox_info["br_y"]}
            """

            print(string_result)

            st.session_state['past'].append("What is my car information:")

            st.session_state['generated'].append(string_result)


        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state['generated'][i], is_user=False, key=str(i), avatar_style= "bottts")

    initialize_session_state()
    display_chat_history()

# if __name__ == "__main__":
main()