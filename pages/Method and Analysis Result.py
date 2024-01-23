import streamlit as st
import pandas as pd

st.title("Method and Analysis Result")

st.write("## Steps in Analysis Method")

st.write("### Data Collection")
st.write("""At this stage, researchers collect data that will be used. In this study, train-clean-100 sample data and train-clean-100 data from librispeech dataset were used.""")

st.write("### Data Processing")
st.write("""At this stage, researchers process the data used. In this study the dataset consists of 80% train data and 20% test data.""")

st.write("### Data Training")
st.write("""At this stage the researcher used a pretraining model for the deep speaker and trained the model for the Gaussian mixture model""")
st.write("""DeepSpeaker is neural speaker embedding system using cosine similarity and train with triplet loss""")
st.write("""Gaussian Mixture Model is probabilistic model that generate with the mixture of gaussian distribution""")

st.write("## Analysis Result")

st.write("### DeepSpeaker")
data_deep = {
    "Dataset": ["train-clean-100-samples", "train-clean-100"],
    "Equal Error Rate": [0.0, 0.0],  # Replace with your actual data
    "Accuracy": [0.998, 0.998]          # Replace with your actual data
}

# Create a DataFrame
df_deep = pd.DataFrame(data_deep)

st.table(df_deep)

st.write("### GMM")
data_gmm = {
    "Dataset": ["train-clean-100-samples", "train-clean-100"],
    "Equal Error Rate": [0.27, 0.3373],  # Replace with your actual data
    "Accuracy": [1, 0.831]          # Replace with your actual data
}

# Create a DataFrame
df_gmm = pd.DataFrame(data_gmm)

st.table(df_gmm)