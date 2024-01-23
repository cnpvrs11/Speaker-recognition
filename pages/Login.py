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

st.title('Login')

# Menentukan direktori utama 'app' untuk mengakses paket 'deep_speaker'
app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(app_dir)
deep_speaker_path = os.path.join(app_dir, 'deep-speaker', 'deep_speaker')

# Menambahkan path 'deep_speaker' ke sys.path
if deep_speaker_path not in sys.path:
    sys.path.append(deep_speaker_path)
# print(sys.path)

from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import SAMPLE_RATE, NUM_FRAMES
from conv_models import DeepSpeakerModel
from Preprocess import batch_cosine_similarity, feature
# from 'test' import batch_cosine_similarity
# from Model import convolutional_model

# Function to validate credentials (placeholder logic)
def validate_credentials(email, password):    
    response = requests.post('http://127.0.0.1:8000/login/', json={'email': email, 'password': password})
    if response.status_code == 200:
        st.success('Upload voice for recognition')
        return True
    else:
        # Menampilkan pesan error dari backend
        # error_detail = response.json().get('detail', 'Login error')
        # st.error(f'Login failed: {error_detail}')
        return False
    # else:
    #     st.error('Login failed. Unknown error.')
    #     return False

# Function to save details in session state
def save_details(email, password, voice_file):
    st.session_state['email'] = email
    st.session_state['password'] = password
    st.session_state['voice_file'] = voice_file
    
def predict(email,password,voice_file):
    # Reproducible results.
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
        
        user_sound_path = sound_paths[0]
        if len(sound_paths)==1:
        # Process each sound path
            mfcc_user = sample_from_mfcc(read_mfcc(os.path.join(f'Sound/{user_id}', user_sound_path), SAMPLE_RATE), NUM_FRAMES)
            mfcc_input = sample_from_mfcc(read_mfcc(voice_file, SAMPLE_RATE), NUM_FRAMES)

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
                st.write('Similarity: ' + str(detected_prob))
            else:
                st.markdown('<h2 style="color: red;">Recognition Failed</h2>', unsafe_allow_html=True)
                st.write('Similarity: ' + str(detected_prob))
        # Record the completion time
            end_time = time.time()
            run_time = end_time - start_time
            st.write('Compiled in ', run_time)
        else:
            st.write('Please use another method')
    else:
        st.error('Failed to get user data.')

def predict_gmm(email,password,voice_file):
    start_time = time.time()

    response = requests.get('http://127.0.0.1:8000/predict/', json={'email': email, 'password': password})

    user_info = response.json()['user_info']

    user_id = user_info['id']
    
    sound_paths = response.json()['sound_paths']
    
    if len(sound_paths)>1:
        features = np.array([])

        for sound in os.listdir('Sound/'+ str(user_id)):
            data_feature=feature('Sound/' + str(user_id) + '/' + sound)
            if len(features)==0:
                features = data_feature
            else:
                features = np.vstack((features,data_feature))
        gmm = mixture.GaussianMixture(n_components=1,covariance_type='diag',n_init=3)
        gmm.fit(features)
        file_path = 'gmm/' + f'gmm_{user_id}.sav'
        pickle.dump(gmm,open(file_path,'wb'))
        
        if os.path.exists(file_path):
        
            gmm_model = pickle.load(open(file_path,'rb'))
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.flac') as temp_file:
                temp_file.write(voice_file.read())
                temp_file_path = temp_file.name
        
            test_feature = feature(temp_file_path)
        
            score_test = gmm_model.score(test_feature)
        
            detected_prob= score_test
            
            if detected_prob>= 0.45:
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
                st.write('Similarity: ' + str(detected_prob))
            else:
                st.markdown('<h2 style="color: red;">Recognition Failed</h2>', unsafe_allow_html=True)
                st.write('Similarity: ' + str(detected_prob))
            os.unlink(temp_file_path)
            end_time = time.time()
            run_time = end_time-start_time
            st.write('Compiled in ',run_time)
        else:
            st.write('Please use another method')
    else:
        st.write('Please use another method')
    
        
# Initialize session states if not already done
if "submit" not in st.session_state:
    st.session_state["submit"] = False

if "view_sound_clicked" not in st.session_state:
    st.session_state["view_sound_clicked"] = False

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

    chosen_method = st.radio("Choose a method:", ['DeepSpeaker', 'GMM'])
    
    col1, col2 = st.columns([7, 1])
    with col1:
        view_sound = st.button('View Sound')
    with col2:
        submit = st.button('Predict')
        
    if voice_file is None:
        st.warning("Please upload your voice file before predict and view sound")
    
    if view_sound and voice_file is not None:
        st.session_state["view_sound_clicked"] = True
        st.write(voice_file.name)
        audio_bytes = voice_file.read()
        st.audio(audio_bytes, format="audio/wav")

    if submit:
        if chosen_method == 'DeepSpeaker':
            if voice_file is not None:
                predict(email, password, voice_file)
        elif chosen_method == 'GMM':
            if voice_file is not None:
                predict_gmm(email, password, voice_file)



