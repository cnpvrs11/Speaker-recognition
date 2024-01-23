import streamlit as st
import os
from pydub import AudioSegment
import io
import requests
import bcrypt
from sklearn import mixture
from Preprocess import feature
st.title('Register')
import pickle
import numpy as np

# Initialize session state variables
if "register" not in st.session_state:
    st.session_state["register"] = False

if "view_register_sound" not in st.session_state:
    st.session_state["view_register_sound"] = False

if "view_register_gmm" not in st.session_state:
    st.session_state["view_register_gmm"] = False

if "register_method" not in st.session_state:
    st.session_state["register_method"] = "DeepSpeaker"

st.markdown("### Select a method for speaker recognition")

methods = ["DeepSpeaker", "GMM"]
st.session_state["register_method"] = st.radio("", methods)

voice_file = None
voice_file_gmm = None

with st.form('register_form'):
    name = st.text_input("Input your name")
    email = st.text_input('Input your email')
    age = st.number_input("Input your age", min_value=0, max_value=100, value=0, step=1)
    gender = st.selectbox("Gender", ("Female", "Male"))
    password = st.text_input('Input Your password', type="password")
    
    if st.session_state["register_method"] == "DeepSpeaker":
        voice_file = st.file_uploader('Choose a File', type=['wav', 'flac'])
    elif st.session_state["register_method"] == "GMM":
        voice_file_gmm = st.file_uploader('Choose a File (Minimal 2 Sound)', accept_multiple_files=True, type=['wav', 'flac'])

    col1, col2 = st.columns([6, 1])
    with col1:
        view_sound = st.form_submit_button('View Sound')
    with col2:
        submit = st.form_submit_button('Register')

    st.warning("Please upload your voice file before register and view sound")
    if view_sound:
        if st.session_state["register_method"] == "DeepSpeaker" and voice_file is not None:
            st.session_state["view_register_sound"] = True
            st.write(voice_file.name)
            audio_bytes = voice_file.read()
            st.audio(audio_bytes, format="audio/wav")
        
        elif st.session_state["register_method"] == "GMM" and voice_file_gmm:
            st.session_state["view_register_gmm"] = True
            for uploaded_file in voice_file_gmm:
                st.write(uploaded_file.name)
                audio_bytes = uploaded_file.read()
                st.audio(audio_bytes, format="audio/wav")
        
    if submit and name and email and password and age and gender:
        st.session_state["register"]= True
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        hashed_password_str = hashed_password.decode('utf-8')
            # Data common to both methods
        user_data = {
            'name': name, 
            'email': email, 
            'age': age, 
            'gender': gender, 
            'password': hashed_password_str
        } 
        
        if st.session_state["register_method"] == "DeepSpeaker" and voice_file:
            user_data['sound_paths'] = [voice_file.name]
        elif st.session_state["register_method"] == "GMM" and voice_file_gmm:
            user_data['sound_paths'] = [file.name for file in voice_file_gmm]
        else:
            st.warning("Please upload your voice file before register")
        response = requests.post('http://127.0.0.1:8000/add_user/', json=user_data)
        if response.status_code == 200:
            response_data = response.json()
            user_id = response_data['user_id']
            save_directory = os.path.join("Sound", str(user_id))

            os.makedirs(save_directory, exist_ok=True)

            if st.session_state["register_method"] == "DeepSpeaker" and voice_file:
                file_path = os.path.join(save_directory, voice_file.name)
                with open(file_path, "wb") as f:
                    f.write(voice_file.getbuffer())
                st.success('Item created successfully!')

            elif st.session_state["register_method"] == "GMM" and voice_file_gmm:
                for uploaded_file in voice_file_gmm:
                    file_path = os.path.join(save_directory, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.success('Item created successfully!')     
            else:
                st.error('Error while creating item.')
        
    elif submit:
        st.error('Please fill all the fields')  
         

        
