import mysql.connector
from fastapi import FastAPI,HTTPException,Form,File,UploadFile,Body
from pydantic import BaseModel
import bcrypt
# import aiofiles
from fastapi.responses import JSONResponse
import os

mydb = mysql.connector.connect(
        host = "localhost",
        user = "root",
        password = "",
        database = "speaker"
)

def initialize_db(mydb=mydb):
    mycursor = mydb.cursor()

    mycursor.execute("""
                 CREATE TABLE IF NOT EXISTS users(
                     id INT AUTO_INCREMENT PRIMARY KEY,
                     name varchar(50) not null,
                     email varchar(50) not null,
                     age int not null,
                     gender varchar(20) not null,
                     password varchar(255) not null,
                     method varchar(20) not null
                 )
                 """)
    
    mycursor.execute("""
        CREATE TABLE IF NOT EXISTS user_sound(
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            sound_path VARCHAR(255) NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    mycursor.execute("""
                     CREATE TABLE IF NOT EXISTS challenge_sound(
                         id INT AUTO_INCREMENT PRIMARY KEY,
                         challenge_category VARCHAR(255),
                         true_path varchar(255),
                         input_path varchar(255)
                     )
                     """)
    
    query = """
    INSERT INTO challenge_sound (challenge_category, true_path, input_path)
    SELECT * FROM (SELECT %s, %s, %s) AS tmp
    WHERE NOT EXISTS (
        SELECT 1 FROM challenge_sound WHERE input_path = %s
    ) LIMIT 1;"""
    
    data_to_insert = [
        ("Sick","silent akhir 3.wav","test-suara6.wav"),
        ("Sick","silent akhir 3.wav","test-suara7.wav"),
        ("Sick","silent akhir 3.wav","test-suara-9.wav"),
        ("Sound too short","silent akhir 3.wav","test-suara.wav"),
        ("Sound too short","silent akhir 3.wav","test-suara10.wav"),
        ("Sound too short","silent akhir 3.wav","test-suara12.wav"),
        ("Noise","silent akhir 3.wav","test-suara5.wav"),
        ("Noise","silent akhir 3.wav","test-suara16.wav"),
        ("Noise","silent akhir 3.wav","test-suara17.wav")
    ]

# Executing the insert command for each data tuple
    for data in data_to_insert:
        mycursor.execute(query, data + (data[2],))  # data[2] is the input_path
        
    mydb.commit()
    mycursor.close()
       
app = FastAPI()

UPLOAD_DIR = "Sound"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    
from typing import List
from fastapi import FastAPI, Body

app = FastAPI()

@app.post("/add_user/")
def create_user(name: str = Body(...), email: str = Body(...), age: int = Body(...), gender: str = Body(...), password: str = Body(...), sound_paths: List[str] = Body(...),method: str = Body(...)):
    conn = mydb
    cursor = conn.cursor()

    # Insert user information
    user_insert_query = """INSERT INTO users (name, email, age, gender, password,method) VALUES (%s, %s, %s, %s, %s,%s)"""
    cursor.execute(user_insert_query, (name, email, age, gender, password,method))
    user_id = cursor.lastrowid  # Assuming you have an auto-increment ID for users

    # Insert sound paths
    sound_path_insert_query = """INSERT INTO user_sound (user_id, sound_path) VALUES (%s, %s)"""
    for sound_path in sound_paths:
        cursor.execute(sound_path_insert_query, (user_id, sound_path))

    conn.commit()
    cursor.close()

    return {"message": "User created successfully.", "user_id": user_id}

    
@app.post("/login")
def login(email:str=Body(...),password:str=Body(...)):
    conn = mydb
    cursor = conn.cursor(dictionary=True)
    user_search_query = """SELECT * FROM users WHERE email = %s"""
    cursor.execute(user_search_query, (email,))
    user = cursor.fetchone()
    cursor.close()
    if user is None:
        raise HTTPException(status_code=404, detail="Email not found")

    # Pastikan password dari database dalam bentuk bytes
    db_password = user['password'].encode('utf-8')

    if bcrypt.checkpw(password.encode('utf-8'), db_password):
        return {"message": "Login Succesfull", "user": user}
    else:
        raise HTTPException(status_code=401, detail="Wrong Password")
    
@app.get("/predict")
def predict(email:str=Body(...),password:str=Body(...)):
    conn = mydb
    cursor = conn.cursor(dictionary=True)
    user_search_query = """SELECT * FROM users WHERE email = %s"""
    cursor.execute(user_search_query, (email,))
    user = cursor.fetchone()
    if user is None:
        raise HTTPException(status_code=404, detail="Email not found")

    user_id = user['id']

    # Retrieve sound paths from the user_sound table
    sound_search_query = """SELECT sound_path FROM user_sound WHERE user_id = %s"""
    cursor.execute(sound_search_query, (user_id,))
    sound_paths = cursor.fetchall()  # This will be a list of dictionaries

    # Close the cursor
    cursor.close()

    # Extract just the sound paths into a list
    sound_paths_list = [sound['sound_path'] for sound in sound_paths]

    # Return the user data along with sound paths
    user_data_with_sound_paths = {
        "user_info": user,
        "sound_paths": sound_paths_list
    }
    return user_data_with_sound_paths
    
@app.get("/list_challenges")
def list_challenge(category: str):
    conn = mydb
    cursor = conn.cursor(dictionary=True)
    challenge_search_query = """SELECT * FROM challenge_sound WHERE challenge_category = %s"""
    cursor.execute(challenge_search_query, (category,))
    data = cursor.fetchall()
    if not data:
        raise HTTPException(status_code=404, detail="Data not found")

    # Close the cursor
    cursor.close()

    # Return the user data along with sound paths
    paths = [row['input_path'] for row in data]
    true_path = [row['true_path'] for row in data]
    
    sound_data = {
        "path" :paths,
        "true" : true_path
    }
    return sound_data
    
@app.get("/list_challenges")
def list_challenge(category: str):
    conn = mydb
    cursor = conn.cursor(dictionary=True)
    challenge_search_query = """SELECT * FROM challenge_sound WHERE challenge_category = %s"""
    cursor.execute(challenge_search_query, (category,))
    data = cursor.fetchall()
    if not data:
        raise HTTPException(status_code=404, detail="Data not found")

    # Close the cursor
    cursor.close()

    # Return the user data along with sound paths
    paths = [row['input_path'] for row in data]
    true_path = [row['true_path'] for row in data]
    
    sound_data = {
        "path" :paths,
        "true" : true_path
    }
    return sound_data

