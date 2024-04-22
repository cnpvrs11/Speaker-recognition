import streamlit as st
import os
import requests
import bcrypt
from Preprocess import feature
import numpy as np
from pydub import AudioSegment
import io
import pickle
from sklearn import mixture
import librosa
import scipy.signal as sps

st.title('Register')

import librosa

def check_audio_duration(audio_file):
    # Membaca file audio
    y, sr = librosa.load(audio_file)
    
    # Menghitung durasi file audio dalam detik
    duration = librosa.get_duration(y=y, sr=sr)
    
    return duration

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
st.write("Register Your Account Before Using Speaker Recognition Application")
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
        view_sound = st.form_submit_button('Listen Sound')
    with col2:
        submit = st.form_submit_button('Register')

    if voice_file is None:
        st.warning("Please upload your voice file before predict and listen sound")
        st.warning("Audio duration must be between 3 and 20 seconds")
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
        user_data = {
            'name': name, 
            'email': email, 
            'age': age, 
            'gender': gender, 
            'password': hashed_password_str,
            'method' : st.session_state["register_method"]
        } 
        
        if st.session_state["register_method"] == "DeepSpeaker" and voice_file:
            duration = check_audio_duration(voice_file)
            if duration >= 3 and duration <= 20:
                user_data['sound_paths'] = [voice_file.name]
            else:
                st.error("Audio file duration must be between 3 and 20 seconds.")
                st.stop()
                
        elif st.session_state["register_method"] == "GMM" and voice_file_gmm:
            all_valid = True
            for uploaded_file in voice_file_gmm:
                duration = check_audio_duration(uploaded_file)
                if duration < 3 or duration > 20:
                    all_valid = False
                    break
            
            if all_valid:
                user_data['sound_paths'] = [file.name for file in voice_file_gmm]
            else:
                st.error("All audio files must have durations between 3 and 20 seconds.")
                st.stop()
                
        else:
            st.warning("Please upload your voice file before register")
            st.stop()
                
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
                st.markdown('<h2 style="color: green;">Register Successfully</h2>', unsafe_allow_html=True)
                st.write('Name: ' + name)
                st.write('Email: ' + email)
                st.write('Age: ' + str(age))
                st.write('Gender: ' + gender)   

            elif st.session_state["register_method"] == "GMM" and voice_file_gmm:
                for uploaded_file in voice_file_gmm:
                    file_path = os.path.join(save_directory, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.markdown('<h2 style="color: green;">Register Successfully</h2>', unsafe_allow_html=True)
                st.write('Name: ' + name)
                st.write('Email: ' + email)
                st.write('Age: ' + str(age))
                st.write('Gender: ' + gender)     
            else:
                st.error('Error while creating item.')
        
    elif submit:
        st.error('Please fill all the fields')
        

        
