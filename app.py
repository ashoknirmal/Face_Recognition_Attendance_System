import cv2
import os
import numpy as np
import pandas as pd
from datetime import date, datetime
from flask import Flask, request, render_template, redirect, url_for, session, flash
from sklearn.neighbors import KNeighborsClassifier
import joblib
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash

# Defining Flask App
app = Flask(__name__)
app.secret_key = 'your_secret_key'

nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Ashok3182004.'
app.config['MYSQL_DB'] = 'face_detection'

# Initialize MySQL Connection
mysql = mysql.connector.connect(
    host=app.config['MYSQL_HOST'],
    user=app.config['MYSQL_USER'],
    password=app.config['MYSQL_PASSWORD'],
    database=app.config['MYSQL_DB']
)
cursor = mysql.cursor()

# Create tables if not exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        user_id INT NOT NULL
    )
""")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS login (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(255) NOT NULL,
        email VARCHAR(255) NOT NULL,
        password VARCHAR(255) NOT NULL
    )
""")
# Table to store attendance data
cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        user_id INT NOT NULL,
        time VARCHAR(255) NOT NULL,
        date DATE NOT NULL
    )
""")

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Directory creation
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# get the number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Insert into CSV file
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')
    
    # Insert into MySQL attendance table
    cursor.execute("INSERT INTO attendance (name, user_id, time, date) VALUES (%s, %s, %s, %s)",
                   (username, userid, current_time, date.today()))
    mysql.commit()

# A function to get names and roll numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l

# A function to delete a user folder 
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)

    user_name, user_id = os.path.basename(duser).split('_')

    cursor.execute("DELETE FROM users WHERE name = %s AND user_id = %s", (user_name, user_id))
    mysql.commit()

    if not os.listdir('static/faces/'):
        os.remove('static/face_recognition_model.pkl')

    try:
        train_model()
    except:
        pass

# ROUTING FUNCTIONS #########################

# Home page (attendance overview)
@app.route('/')
def home():
    if 'loggedin' in session:
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)
    else:
        return redirect(url_for('login'))

# List users page
@app.route('/listusers')
def listusers():
    if 'loggedin' in session:
        userlist, names, rolls, l = getallusers()
        return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)
    else:
        return redirect(url_for('login'))

# Delete user
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    if 'loggedin' in session:
        duser = request.args.get('user')
        deletefolder('static/faces/'+duser)

        userlist, names, rolls, l = getallusers()
        return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)
    else:
        return redirect(url_for('login'))

# Face recognition attendance functionality
@app.route('/start', methods=['GET'])
def start():
    if 'loggedin' in session:
        names, rolls, times, l = extract_attendance()

        if 'face_recognition_model.pkl' not in os.listdir('static'):
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model. Please add a new face to continue.')

        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if len(extract_faces(frame)) > 0:
                (x, y, w, h) = extract_faces(frame)[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
                cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]
                add_attendance(identified_person)
                cv2.putText(frame, f'{identified_person}', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)
    else:
        return redirect(url_for('login'))

# Add new user page
@app.route('/add', methods=['GET', 'POST'])
def add():
    if 'loggedin' in session:
        if request.method == 'POST':
            name = request.form['newusername']
            roll = request.form['newuserid']

            userimagefolder = f'static/faces/{name}_{roll}'
            if not os.path.isdir(userimagefolder):
                os.makedirs(userimagefolder)
            
            cap = cv2.VideoCapture(0)
            i, j = 0, 0
            while True:
                ret, frame = cap.read()
                if len(extract_faces(frame)) > 0:
                    (x, y, w, h) = extract_faces(frame)[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
                    cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
                    face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                    if j % 10 == 0:
                        cv2.imwrite(f'{userimagefolder}/{i}.jpg', face)
                        i += 1
                    j += 1
                    cv2.putText(frame, f'{str(nimgs-i)} images left', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Adding new User', frame)
                if cv2.waitKey(1) == 27 or i == nimgs:
                    break
            cap.release()
            cv2.destroyAllWindows()

            train_model()

            # Insert the new user into MySQL database
            cursor.execute("INSERT INTO users (name, user_id) VALUES (%s, %s)", (name, roll))
            mysql.commit()

            names, rolls, times, l = extract_attendance()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)
        else:
            return render_template('add.html', totalreg=totalreg(), datetoday2=datetoday2)
    else:
        return redirect(url_for('login'))

# Register page
@app.route('/signup', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['name']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        cursor.execute("INSERT INTO login (username, email, password) VALUES (%s, %s, %s)", (username, email, hashed_password))
        mysql.commit()

        flash('You are successfully registered! Please login.')
        return redirect(url_for('login'))
    return render_template('signup.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cursor.execute("SELECT * FROM login WHERE email = %s", (email,))
        account = cursor.fetchone()

        if account and check_password_hash(account[3], password):
            session['loggedin'] = True
            session['id'] = account[0]
            session['email'] = account[2]

            return redirect(url_for('home'))
        else:
            flash('Incorrect email/password!')
            return redirect(url_for('login'))
    return render_template('login.html')

# Logout functionality
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Running the app
if __name__ == '__main__':
    app.run(debug=True)
