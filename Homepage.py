import streamlit as st
import sklearn 
from Database import initialize_db
st.set_page_config(
    page_title="Speaker Recognition App"
)

st.title('Speaker Recognition Application')

initialize_db()

st.write("""## How to use Speaker Recognition Application""")
st.write("""1. Register your Sound and Data""")
st.write("""2. Login with Your Email and Password""")
st.write("""3. Upload your voice to recognition""")
st.write("""4. Successfully Recognition""")
st.image("landing.jpg")




