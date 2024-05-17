import streamlit as st
import os
import mysql.connector
import requests
import bcrypt
import sys
import random
import time
import numpy as np
import pickle
import tempfile
from sklearn import mixture
import math
import soundfile as sf

st.title('Login')
st.write('Input Your Email and Password To Do Speaker Recognition')

# Menentukan direktori utama 'app' untuk mengakses paket 'deep_speaker'
app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
deep_speaker_path = os.path.join(app_dir, 'deep-speaker', 'deep_speaker')

# Menambahkan path 'deep_speaker' ke sys.path
if deep_speaker_path not in sys.path:
    sys.path.append(deep_speaker_path)

from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import SAMPLE_RATE, NUM_FRAMES
from conv_models import DeepSpeakerModel
from Preprocess import batch_cosine_similarity, feature

# Function to validate credentials
def validate_credentials(email, password):    
    response = requests.post('http://127.0.0.1:8000/login/', json={'email': email, 'password': password})
    if response.status_code == 200:
        st.success('Upload voice for recognition')
        return True
    else:
        return False
    
def get_method(email,password):
    response = requests.post('http://127.0.0.1:8000/login/', json={'email': email, 'password': password})
    if response.status_code == 200:
        data = response.json()['user']
        method = data['method']
        return method
        
# Function to save details in session state
def save_details(email, password, voice_file):
    st.session_state['email'] = email
    st.session_state['password'] = password
    st.session_state['voice_file'] = voice_file
    
def predict(email,password,voice_file):
    # Reproducible results.z
    audio_bytes = voice_file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_file.write(audio_bytes)
        temp_file_path = temp_file.name

        # Open the temporary WAV file with soundfile
        with sf.SoundFile(temp_file_path) as sf_desc:
            duration = len(sf_desc) / sf_desc.samplerate

    st.session_state["duration"]=duration
    if duration < 3 or duration > 20:
        st.error("Audio duration must be between 3 and 20 seconds.")
        st.stop()
    start_time = time.time()
    response = requests.get('http://127.0.0.1:8000/predict/', json={'email': email, 'password': password})
    if response.status_code == 200:
        user_info = response.json()['user_info']
        sound_paths = response.json()['sound_paths']
        
        user_id = user_info['id']
        np.random.seed(123)
        random.seed(123)
    
        # Define and load the model
        model = DeepSpeakerModel()
        model.m.load_weights('ResCNN_triplet_training_checkpoint_265.h5', by_name=True)
        
        if len(sound_paths)==1:
            for user_sound_path in sound_paths:
                print(user_sound_path)
        # Process each sound path
                mfcc_user = sample_from_mfcc(read_mfcc(os.path.join(f'Sound/{user_id}', user_sound_path), SAMPLE_RATE), NUM_FRAMES)
                mfcc_input = sample_from_mfcc(read_mfcc(temp_file_path, SAMPLE_RATE), NUM_FRAMES)

                # Call the model to get the embeddings for each file
                predict_user = model.m.predict(np.expand_dims(mfcc_user, axis=0))
                predict_input = model.m.predict(np.expand_dims(mfcc_input, axis=0))

                # Calculate similarity using cosine similarity
                detected_prob = batch_cosine_similarity(predict_user, predict_input)

                if detected_prob >= 0.45:
                    st.session_state["predict"] = True
                    st.markdown('<h2 style="color: green;">Recognition Successfully</h2>', unsafe_allow_html=True)
                    st.write('Name: ' + user_info['name'])
                    st.write('Email: ' + user_info['email'])
                    st.write('Age: ' + str(user_info['age']))
                    st.write('Gender: ' + user_info['gender'])
                    st.write('Similarity: ' + str(detected_prob*100) +'%')
                else:
                    st.markdown('<h2 style="color: red;">Recognition Failed</h2>', unsafe_allow_html=True)
                    st.write('Similarity: ' + str(detected_prob*100) +'%')
        # Record the completion time
            end_time = time.time()
            run_time = end_time - start_time
            st.write('Compiled in '+ str(run_time) +' second')
            os.unlink(temp_file_path)
        else:
            st.write('Please use another method')
    else:
        st.error('Failed to get user data.')

def predict_gmm(email,password,voice_file):
    audio_bytes = voice_file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_file.write(audio_bytes)
        temp_file_path = temp_file.name

        # Open the temporary WAV file with soundfile
        with sf.SoundFile(temp_file_path) as sf_desc:
            duration = len(sf_desc) / sf_desc.samplerate

    st.session_state["duration"]=duration
    if duration < 3 or duration > 20:
        st.error("Audio duration must be between 3 and 20 seconds.")
        st.stop()
    start_time = time.time()

    response = requests.get('http://127.0.0.1:8000/predict/', json={'email': email, 'password': password})

    user_info = response.json()['user_info']

    user_id = user_info['id']
    
    sound_paths = response.json()['sound_paths']
    
    if len(sound_paths) > 1:
        features = np.asarray(())  # Initialize an empty array with shape (0, 52)

        for sound in os.listdir('Sound/' + str(user_id)):
            data_feature = feature('Sound/' + str(user_id) + '/' + sound)
            if len(features)==0:
                features= data_feature
            else:
                features = np.vstack((features, data_feature))
        
        gmm = mixture.GaussianMixture(n_components=2, covariance_type='diag')
        gmm.fit(features)
        
        os.makedirs('gmm')
        
        file_path = 'gmm/' + f'gmm_{user_id}.sav'
        pickle.dump(gmm, open(file_path, 'wb'))
    
        gmm_files = [os.path.join('gmm', fname) for fname in os.listdir('gmm') if fname.endswith('.sav')]

        # Membaca model GMM dari file
        gmm_models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]

        # Memperbaiki split dengan os.sep
        unique_speakers = [fname.split(os.sep)[-1].split(".gmm")[0] for fname in gmm_files]
        
        log_likelihood = np.zeros(len(gmm_models))
        
        test_feature = feature(temp_file_path)
        for i,gmm in enumerate(gmm_models):
            score_test = gmm.score(test_feature)
            log_likelihood[i] = score_test.sum()
            
        y_pred = np.argmax(log_likelihood)
        true_speaker = f'gmm_{user_id}.sav'
        predicted_speaker = unique_speakers[y_pred]
        if true_speaker==predicted_speaker:
            st.session_state["predict"] = True
            name = user_info['name']
            email = user_info['email']
            age = user_info['age']
            gender = user_info['gender']
            st.markdown('<h2 style="color: green;">Recognition Successfully</h2>', unsafe_allow_html=True)
            st.write('Name: ' + name)
            st.write('Email: ' + email)
            st.write('Age: ' + str(age))
            st.write('Gender: ' + gender)
        else:
            st.markdown('<h2 style="color: red;">Recognition Failed</h2>', unsafe_allow_html=True)
        os.unlink(temp_file_path)
        end_time = time.time()
        run_time = end_time-start_time
        st.write('Compiled in '+str(run_time) + ' second')
    else:
        st.write('Please use another method')
            
# Initialize session states if not already done
if "submit" not in st.session_state:
    st.session_state["submit"] = False

if "view_sound_clicked" not in st.session_state:
    st.session_state["view_sound_clicked"] = False

if "duration" not in st.session_state:
    st.session_state["duration"]=False

# User inputs for email and password
with st.form("login_form"):
    email = st.text_input('Please input your email')    
    password = st.text_input('Your password', type="password")
    login_submit = st.form_submit_button("Login")

# Check if login is submitted and credentials are valid
if login_submit and email and password and validate_credentials(email, password):
    st.session_state["submit"] = True
elif login_submit and email and password:
    st.error("Invalid email or password.")
elif login_submit:
    st.error("Please fill all the fields")

if st.session_state["submit"]:
    # File uploader and buttons are displayed only after successful login
    voice_file = st.file_uploader('Choose a File', key='voice_file_uploader',type=['wav','flac'])
    
    col1, col2 = st.columns([7, 1])
    with col1:
        view_sound = st.button('Listen Sound')
    with col2:
        submit = st.button('Predict')
        
    if voice_file is None:
        st.warning("Please upload your voice file before predict and listen sound")
        st.warning("Audio duration must be between 3 and 20 seconds")
    
    if view_sound and voice_file is not None:
        st.session_state["view_sound_clicked"] = True
        st.write(voice_file.name)
        audio_bytes = voice_file.read()
        st.audio(audio_bytes, format="audio/wav")

    chosen_method = get_method(email,password)
    if submit and voice_file is not None:
        if chosen_method == 'DeepSpeaker':
            predict(email, password, voice_file)
        elif chosen_method == 'GMM':
            predict_gmm(email, password, voice_file)



