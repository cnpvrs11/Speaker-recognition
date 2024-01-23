import streamlit as st
import requests

st.title("Challenge in Recognition")

st.write("This is the example of challenge in speaker recognition")

category = st.selectbox("Challenge", ("Sound too short", "Noise", "Sick"))

if category:
    response = requests.get('http://127.0.0.1:8000/list_challenges/', params={'category': category})
    if response.status_code == 200:
        sound_data = response.json()
        paths = sound_data["path"]
        true_paths = sound_data["true"]

        # Assuming you want to create a dropdown to select an input path
        selected_index = st.selectbox("Select Input", range(len(paths)))
        selected_path = paths[selected_index]
        selected_true_path = true_paths[selected_index]

        view = st.button("View Sound")
    
        if view:
            st.write("True Path")
            st.audio(f"challenge/{selected_true_path}", format="audio/wav")
            
            st.write("Input Path")
            st.audio(f"challenge/{selected_path}", format="audio/wav")
    else:
        st.error("Error fetching data")
    
    
    