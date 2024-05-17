# Speaker Recognition Application

Speaker Recognition Application is a web application designed to identify users logging into the system. It employs a two-factor authentication process: first, users log in with their email and password, and then they upload a voice sample for recognition.  Developed using Streamlit and Python, this application utilizes the **Deep Speaker** model and **Gaussian Mixture Models** for the speaker recognition process.
![image](https://github.com/cnpvrs11/Speaker-recognition/assets/85270958/3ce26cbb-ba00-4043-8582-695e10b55f97)


## Installation
Please install in the following order

1. Make Virtual Environment in Terminal
```bash
virtualenv venv
```
2. Activate Virtual Environment
```bash
venv\Scripts\activate
```
3. Install Library
```bash
pip install tensorflow
pip install streamlit
pip install tensorboard==2.11.2
pip install protobuf
pip install scikit-learn
pip install mysql-connector-python
pip install bcrypt
pip install soundfile
pip install python-speech-features
pip install tqdm
pip install dill
pip install natsort
pip install pyAudioAnalysis
pip install matplotlib
pip install eyed3
pip install pydub
pip install fastapi
pip install librosa==0.10.1
```
## Run Application on Local Environment
After installing all the required libraries, you can run this application on your local machine by running this command. Please make sure that you add **streamlit** command to your PATH environment variable.

```bash
streamlit run Homepage.py
```

## Run FastAPI on Local Environment

```bash
uvicorn Database:app --reload
```

Don't forget to run your XAMPP or other database machine for local database
