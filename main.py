import streamlit as st
from streamlit_mic_recorder import mic_recorder

import cv2
import numpy as np

import io
import os
import random
import time
import base64

from anthropic import Anthropic
from openai import OpenAI

from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, AIMessage
from langchain.chains import LLMChain

# Set your API keys
os.environ['ANTHROPIC_API_KEY'] = st.secrets["ANTHROPIC_API_KEY"]
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['GROQ_API_KEY'] = st.secrets["GROQ_API_KEY"]

client = OpenAI()

### ---- Helper functions ---- ###
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Please share your thoughts to start."}]
    
def get_base64_encoded_image(image_bytes):
    base_64_encoded_data = base64.b64encode(image_bytes)
    base64_string = base_64_encoded_data.decode('utf-8')
    return base64_string

def process_image_with_anthropic(image_bytes):
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    MODEL_NAME = "claude-3-opus-20240229"
    
    base64_image = get_base64_encoded_image(image_bytes)
    message_list = [
        {
            "role": 'user',
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}},
                {"type": "text", "text": "Describe the content of this image."}
            ]
        }
    ]

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=message_list
    )
    return response.content[0].text

def transcribe_audio(audio_bytes):
    # Using OpenAI Whisper for transcription
    audio_bio = io.BytesIO(audio_bytes)
    audio_bio.name = 'audio.wav'
    transcript = client.audio.transcriptions.create(model="whisper-1", 
                                         file=audio_bio, language="en")
    return transcript.text

def sample_string():
    # fun responses
    responses = ["I'm not sure what you mean by that.",
                 "I am pickle rickkkkk",
                 "Loved the autists"]
    
    sampled = random.choice(responses)
    return sampled

def llm_response(chat_history):
    
    if st.session_state.image_description is not None:
        ai_message = f"Image description: {st.session_state.image_description}"
    else:
        ai_message = "Image description: No picture provided. Ignore this message."
    
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are a fun pirate who loves sharing jokes and short stories from interesting adventures. The user might share a picture with you, make sure to read the description and respond accordingly. If no pictures have been shared, ignore the image description."),
            
            AIMessage(content=ai_message),
            
            HumanMessagePromptTemplate.from_template("{chat_history}"),
        ]
    )

    llm = ChatGroq(model_name="llama3-70b-8192", 
                   temperature=0.5, max_tokens=250)
    
    #llm = ChatAnthropic(model='claude-3-opus-20240229',
    #                    temperature=0.5, max_tokens=250)
    
    chat_llm_chain = LLMChain(llm=llm, prompt=prompt)
    response = chat_llm_chain.predict(chat_history=chat_history)
    
    # Reset the image description after generating the response
    st.session_state.image_description = None
    
    return response
    

def llm_response_generator(llm_response):
    """
    Generator function that yields formatted parts of the LLM response.
    It splits the response into paragraphs using newline characters and yields each paragraph.
    This helps in maintaining the structured output for display in the Streamlit chat interface.
    """
    paragraphs = llm_response.split('\n')  # Split on double newline for distinct sections
    for paragraph in paragraphs:
        if paragraph.strip():  # Only yield non-empty paragraphs
            yield paragraph
            time.sleep(0.05)
        yield '\n'
### ------------------------- ###

def chat_main():
    st.title("Novem 2.0 - BETA")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "last_image_processed" not in st.session_state:
        st.session_state.last_image_processed = None
        
    if "image_description" not in st.session_state:
        st.session_state.image_description = None
        
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sidebar for microphone input
    with st.sidebar:
        st.markdown("### **Talk to Novem:**")
        audio = mic_recorder(start_prompt="Start recording", 
                             stop_prompt="Stop recording", 
                             key='recorder',
                             format="wav",
                             just_once=True)
        
        st.markdown("### **Share a picture:**")
        img_file_buffer = st.camera_input("Take the picture")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 
                                   cv2.IMREAD_COLOR)
            
            if st.session_state.last_image_processed != bytes_data:
                st.write("### **Captured Image**")
                st.image(cv2_img, channels="BGR")
                st.write(cv2_img.shape)
                
                with st.spinner('Processing image...'):
                    image_description = process_image_with_anthropic(bytes_data)
                    st.session_state.image_description = image_description
                    
                    st.session_state.last_image_processed = bytes_data
                    st.session_state.messages.append({"role": "user", "content": f"Image description: {image_description}"})
                    with st.chat_message("user"):
                        st.write(f"Image description: {image_description}")
                        

    if audio is not None:
        audio_bytes = audio['bytes']
        st.sidebar.audio(audio_bytes,
                         format="audio/mpeg")

        # Transcribe audio input
        with st.spinner('Transcribing...'):
            audio_user_input = transcribe_audio(audio_bytes)
            st.sidebar.write(f"Transcribed text: {audio_user_input}")

            # Append the transcribed text to the chat history
            st.session_state.messages.append({"role": "user", 
                                        "content": audio_user_input})
            
            with st.chat_message("user"):
                st.write(audio_user_input)

            ## assistant_response = sample_string()
            llm_audio_res = llm_response(st.session_state.messages)
            st.session_state.messages.append({"role": "Novem", 
                                            "content": llm_audio_res})
            
            with st.chat_message("Novem"):
                st.write_stream(llm_response_generator(llm_audio_res))
                #print(type(llm_audio_res))
                
            audio = None
            
    if text_user_input := st.chat_input("Type a message..."):
        
        # append user text input to chat history
        st.session_state.messages.append({"role": "user", 
                                        "content": text_user_input})
        
        with st.chat_message("user"):
            st.write(text_user_input)
            
        with st.chat_message("Novem"):
            with st.spinner('Thinking...'):
                ## assistant_response = sample_string()
                llm_text_res = llm_response(st.session_state.messages)
                #print(type(llm_text_res))
                
                st.write_stream(llm_response_generator(llm_text_res))
                
        st.session_state.messages.append({"role": "Novem", 
                                            "content": llm_text_res})
        
    #print(st.session_state.messages)

if __name__ == "__main__":
    chat_main()
