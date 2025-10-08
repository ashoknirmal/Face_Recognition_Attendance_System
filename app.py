import cv2
import os
import numpy as np
import pandas as pd
from datetime import date, datetime
from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
from sklearn.neighbors import KNeighborsClassifier
import joblib
import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import shutil
from authlib.integrations.flask_client import OAuth
import requests

# Load environment variables
load_dotenv()

# Defining Flask App
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'fallback_secret_key')

# Initialize OAuth
oauth = OAuth(app)

# Google OAuth2 Configuration (Updated for 2025)
app.config['OAUTH2_PROVIDERS'] = {
    'google': {
        'client_id': os.getenv('GOOGLE_CLIENT_ID'),
        'client_secret': os.getenv('GOOGLE_CLIENT_SECRET'),
        'authorize_url': 'https://accounts.google.com/o/oauth2/v2/auth',
        'access_token_url': 'https://oauth2.googleapis.com/token',
        'api_base_url': 'https://www.googleapis.com/oauth2/v2/',
        'client_kwargs': {
            'scope': 'openid email profile'
        },
        'userinfo_endpoint': 'https://openidconnect.googleapis.com/v1/userinfo',
        'server_metadata_url': 'https://accounts.google.com/.well-known/openid-configuration'
    }
}

# Register Google provider
google = oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    authorize_url='https://accounts.google.com/o/oauth2/v2/auth',
    access_token_url='https://oauth2.googleapis.com/token',
    api_base_url='https://www.googleapis.com/oauth2/v2/',
    client_kwargs={'scope': 'openid email profile'},
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration'
)

nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# MySQL Configuration
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST', 'localhost')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', 'Ashok3182004.')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB', 'face_detection')

def get_db_connection():
    """Get MySQL database connection"""
    try:
        connection = mysql.connector.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            database=app.config['MYSQL_DB']
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def safe_makedirs(path, verbose=True):
    """Safely create directories, handling conflicts"""
    try:
        if os.path.isfile(path):
            if verbose:
                print(f"⚠️  Removing conflicting file: {path}")
            os.remove(path)
        
        if os.path.exists(path) and os.path.isdir(path):
            if verbose:
                print(f"✅ Directory already exists: {path}")
            return True
        
        os.makedirs(path, exist_ok=True)
        if verbose:
            print(f"✅ Created directory: {path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating directory {path}: {e}")
        return False

def init_directories():
    """Initialize all required directories safely"""
    print("=== Initializing directories ===")
    
    dirs_to_create = [
        'Attendance',
        'static',
        'static/faces'
    ]
    
    success = True
    for dir_path in dirs_to_create:
        if not safe_makedirs(dir_path):
            success = False
    
    csv_path = f'Attendance/Attendance-{datetoday}.csv'
    if not os.path.exists(csv_path):
        try:
            with open(csv_path, 'w') as f:
                f.write('Name,Roll,Time\n')
            print(f"✅ Created attendance CSV: {csv_path}")
        except Exception as e:
            print(f"❌ Error creating CSV: {e}")
            success = False
    else:
        print(f"✅ Attendance CSV exists: {csv_path}")
    
    print(f"Directory setup: {'✅ SUCCESS' if success else '❌ FAILED'}")
    return success

def init_db():
    """Initialize database tables"""
    connection = get_db_connection()
    if connection is None:
        print("❌ Failed to connect to database for initialization")
        return False
    
    try:
        cursor = connection.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS login (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) NOT NULL UNIQUE,
                email VARCHAR(255) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                user_id VARCHAR(20) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_user (name, user_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                user_id VARCHAR(20) NOT NULL,
                time TIME NOT NULL,
                date DATE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_attendance (name, user_id, date),
                KEY idx_user_date (user_id, date)
            )
        """)
        
        connection.commit()
        print("✅ Database tables initialized successfully")
        return True
        
    except Error as e:
        print(f"❌ Error initializing database: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def init_face_detector():
    """Initialize face detector with error handling"""
    try:
        if not os.path.exists('haarcascade_frontalface_default.xml'):
            print("❌ ERROR: haarcascade_frontalface_default.xml not found!")
            print("Please download it from: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml")
            return None
        
        detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        if detector.empty():
            print("❌ ERROR: Failed to load haarcascade file!")
            return None
        
        print("✅ Face detector initialized successfully")
        return detector
        
    except Exception as e:
        print(f"❌ Error initializing face detector: {e}")
        return None

# Initialize everything at startup
print("🚀 Starting Face Recognition Attendance System...")
print(f"📅 Today's date: {datetoday2}")

db_success = init_db()
dir_success = init_directories()
face_detector = init_face_detector()

def totalreg():
    """Get total number of registered users"""
    try:
        faces_dir = 'static/faces'
        if not os.path.exists(faces_dir) or not os.path.isdir(faces_dir):
            return 0
        return len([d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))])
    except:
        return 0

def extract_faces(img):
    """Extract faces from image using Haar cascade"""
    global face_detector
    try:
        if face_detector is None:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except Exception as e:
        print(f"Error in face extraction: {e}")
        return []

def identify_face(facearray):
    """Identify face using trained model"""
    try:
        if not os.path.exists('static/face_recognition_model.pkl'):
            print("⚠️  No trained model found. Please add users first.")
            return None
        model = joblib.load('static/face_recognition_model.pkl')
        prediction = model.predict(facearray)[0]
        return prediction
    except Exception as e:
        print(f"Face recognition error: {e}")
        return None

def train_model():
    """Train KNN model on all available faces"""
    try:
        faces_dir = 'static/faces'
        if not os.path.exists(faces_dir):
            print("⚠️  No faces directory found")
            return False
        
        faces = []
        labels = []
        userlist = os.listdir(faces_dir)
        
        total_images = 0
        for user in userlist:
            user_path = os.path.join(faces_dir, user)
            if os.path.isdir(user_path):
                for imgname in os.listdir(user_path):
                    img_path = os.path.join(user_path, imgname)
                    img = cv2.imread(img_path)
                    if img is not None:
                        resized_face = cv2.resize(img, (50, 50))
                        faces.append(resized_face.ravel())
                        labels.append(user)
                        total_images += 1
        
        if len(faces) > 0:
            faces = np.array(faces)
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(faces, labels)
            joblib.dump(knn, 'static/face_recognition_model.pkl')
            print(f"✅ Model trained with {total_images} face samples from {len(userlist)} users")
            return True
        else:
            print("⚠️  No face data available for training")
            return False
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False

def extract_attendance():
    """Extract attendance data from CSV"""
    try:
        csv_path = f'Attendance/Attendance-{datetoday}.csv'
        if not os.path.exists(csv_path):
            return [], [], [], 0
        
        df = pd.read_csv(csv_path)
        names = df['Name'].tolist()
        rolls = df['Roll'].tolist()
        times = df['Time'].tolist()
        l = len(df)
        return names, rolls, times, l
    except FileNotFoundError:
        return [], [], [], 0
    except Exception as e:
        print(f"Error reading attendance file: {e}")
        return [], [], [], 0

def add_attendance(name):
    """Add attendance for a user"""
    try:
        username = name.split('_')[0]
        userid = name.split('_')[1]
        current_time = datetime.now().strftime("%H:%M:%S")
        
        csv_path = f'Attendance/Attendance-{datetoday}.csv'
        df = pd.read_csv(csv_path)
        
        if int(userid) not in list(df['Roll']):
            new_record = pd.DataFrame({
                'Name': [name],
                'Roll': [userid],
                'Time': [current_time]
            })
            df = pd.concat([df, new_record], ignore_index=True)
            df.to_csv(csv_path, index=False)
            
            connection = get_db_connection()
            if connection:
                try:
                    cursor = connection.cursor()
                    cursor.execute("""
                        INSERT INTO attendance (name, user_id, time, date) 
                        VALUES (%s, %s, %s, %s)
                    """, (username, userid, current_time, date.today()))
                    connection.commit()
                except Error as e:
                    print(f"Database insert error: {e}")
                finally:
                    cursor.close()
                    connection.close()
            
            print(f"✅ Attendance marked for {name} at {current_time}")
            return True
        else:
            print(f"⚠️  Attendance already marked for {name}")
            return False
    except Exception as e:
        print(f"❌ Error adding attendance: {e}")
        return False

def get_or_create_user_from_google(user_info):
    """Get or create user based on Google profile"""
    email = user_info.get('email')
    username = user_info.get('name', email.split('@')[0])
    
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM login WHERE email = %s", (email,))
            account = cursor.fetchone()
            
            if account:
                return account
            else:
                hashed_password = generate_password_hash('oauth_google')
                cursor.execute("""
                    INSERT INTO login (username, email, password) 
                    VALUES (%s, %s, %s)
                """, (username, email, hashed_password))
                connection.commit()
                cursor.execute("SELECT * FROM login WHERE email = %s", (email,))
                return cursor.fetchone()
        except Error as e:
            print(f"Error handling Google user: {e}")
            flash('Error processing Google login', 'error')
        finally:
            cursor.close()
            connection.close()
    return None

# Routes
@app.route('/')
@app.route('/home')
def home():
    """Home page - requires login"""
    if 'loggedin' not in session:
        flash('Please login to access this page', 'warning')
        return redirect(url_for('login'))
    
    names, rolls, times, l = extract_attendance()
    total_users = totalreg()
    return render_template('home.html', 
                         names=names, 
                         rolls=rolls, 
                         times=times, 
                         l=l, 
                         totalreg=total_users,
                         datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    """Start attendance marking and stop camera after marking or checking attendance"""
    if 'loggedin' not in session:
        flash('Please login to access this page', 'warning')
        return redirect(url_for('login'))
    
    cap = None
    processed_users = set()  # Track processed users in this session
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            flash('Cannot access webcam. Please check camera permissions.')
            return redirect(url_for('home'))
        
        print("🎥 Starting attendance mode - Press ESC to exit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            faces = extract_faces(frame)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 2)
                cv2.rectangle(frame, (x, y-40), (x+w, y), (86, 32, 251), -1)
                
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                face_array = face.reshape(1, -1)
                
                identified_person = identify_face(face_array)
                if identified_person and identified_person not in processed_users:
                    processed_users.add(identified_person)
                    if add_attendance(identified_person):
                        cv2.putText(frame, f'ID: {identified_person}', 
                                  (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, 'Marked!', (x+5, y+h+25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow('Attendance System - Press ESC to exit', frame)
                        cv2.waitKey(1000)  # Display for 1 second
                        flash(f'Marked as present for {identified_person}', 'success')
                        break  # Stop camera after successful marking
                    else:
                        cv2.putText(frame, f'ID: {identified_person}', 
                                  (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, 'Already Marked', (x+5, y+h+25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow('Attendance System - Press ESC to exit', frame)
                        cv2.waitKey(1000)  # Display for 1 second
                        flash(f'Attendance already marked for {identified_person}', 'warning')
                        break  # Stop camera after checking attendance
                elif not identified_person:
                    cv2.putText(frame, 'Unknown', (x+5, y-15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Attendance System - Press ESC to exit', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break
                
    except Exception as e:
        print(f"❌ Error in attendance marking: {e}")
        flash('Error during attendance marking', 'error')
    
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
    
    names, rolls, times, l = extract_attendance()
    total_users = totalreg()
    return render_template('home.html', 
                         names=names, 
                         rolls=rolls, 
                         times=times, 
                         l=l, 
                         totalreg=total_users,
                         datetoday2=datetoday2)

@app.route('/add', methods=['GET', 'POST'])
def add():
    """Add new user"""
    if 'loggedin' not in session:
        flash('Please login to access this page', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        newusername = request.form['newusername'].strip()
        newuserid = request.form['newuserid'].strip()
        
        if not newusername or not newuserid:
            flash('Please fill both username and user ID', 'error')
            return render_template('add.html', totalreg=totalreg(), datetoday2=datetoday2)
        
        userimagefolder = f'static/faces/{newusername}_{newuserid}'
        
        try:
            if not os.path.isdir(userimagefolder):
                os.makedirs(userimagefolder)
            
            i, j = 0, 0
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                flash('Cannot access webcam. Please check camera permissions.')
                return render_template('add.html', totalreg=totalreg(), datetoday2=datetoday2)
            
            print(f"📸 Capturing {nimgs} images for {newusername} (ID: {newuserid})")
            print("Press ESC to stop capturing")
            
            while i < nimgs:
                ret, frame = cap.read()
                if not ret:
                    break
                
                faces = extract_faces(frame)
                
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                    cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                    
                    if j % 10 == 0:
                        name = f'{newusername}_{i}.jpg'
                        cv2.imwrite(os.path.join(userimagefolder, name), frame[y:y+h, x:x+w])
                        i += 1
                        print(f"📷 Captured image {i}/{nimgs}")
                
                j += 1
                cv2.imshow('Adding new User - Press ESC to finish', frame)
                
                if cv2.waitKey(1) == 27 or i == nimgs:
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            if train_model():
                flash(f'User {newusername} added and model trained successfully!', 'success')
            else:
                flash(f'User {newusername} added but model training failed. Please add more users.', 'warning')
            
            connection = get_db_connection()
            if connection:
                try:
                    cursor = connection.cursor()
                    cursor.execute("INSERT INTO users (name, user_id) VALUES (%s, %s)", 
                                 (newusername, newuserid))
                    connection.commit()
                    print(f"✅ User {newusername} added to database")
                except Error as e:
                    print(f"⚠️  Error adding user to db: {e}")
                    flash(f'User {newusername} added but database error occurred', 'warning')
                finally:
                    cursor.close()
                    connection.close()
            
        except Exception as e:
            print(f"❌ Error adding user: {e}")
            flash('Error adding user. Please try again.', 'error')
        
        names, rolls, times, l = extract_attendance()
        total_users = totalreg()
        return render_template('home.html',
                             names=names,
                             rolls=rolls,
                             times=times,
                             l=l,
                             totalreg=total_users,
                             datetoday2=datetoday2)
    
    return render_template('add.html', totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/logout')
def logout():
    """User logout"""
    if 'username' in session:
        flash(f'Goodbye, {session["username"]}!', 'info')
    session.clear()
    return redirect(url_for('login'))

@app.route('/auth/google')
def google_login():
    """Initiate Google OAuth login"""
    if not os.getenv('GOOGLE_CLIENT_ID') or not os.getenv('GOOGLE_CLIENT_SECRET'):
        flash('Google authentication not configured. Please use email login.', 'error')
        return redirect(url_for('login'))
    
    try:
        # Debug: Test OpenID configuration endpoint
        response = requests.get('https://accounts.google.com/.well-known/openid-configuration')
        response.raise_for_status()
        print("✅ Google OpenID configuration fetched successfully")
        
        redirect_uri = url_for('google_callback', _external=True)
        return google.authorize_redirect(redirect_uri)
    except requests.exceptions.HTTPError as e:
        print(f"❌ Error fetching Google OpenID configuration: {e}")
        flash(f'Google authentication failed: {str(e)}. Please use email login.', 'error')
        return redirect(url_for('login'))
    except Exception as e:
        print(f"❌ Error initiating Google OAuth: {e}")
        flash('Failed to initiate Google authentication. Please try email login.', 'error')
        return redirect(url_for('login'))

@app.route('/auth/callback/google')
def google_callback():
    """Handle Google OAuth callback"""
    try:
        token = google.authorize_access_token()
        user_info = google.get('userinfo').json()
        
        account = get_or_create_user_from_google(user_info)
        if account:
            session['loggedin'] = True
            session['id'] = account[0]
            session['username'] = account[1]
            session['email'] = account[2]
            flash(f'Welcome back, {account[1]}! (Google Login)', 'success')
            return redirect(url_for('home'))
        else:
            flash('Failed to authenticate with Google', 'error')
            return redirect(url_for('login'))
    except Exception as e:
        print(f"❌ Google callback error: {e}")
        flash('Google authentication failed. Please try again or use email login.', 'error')
        return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def register():
    """User registration (traditional email/password)"""
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password'].strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        
        if not all([username, email, password]):
            flash('Please fill all fields', 'error')
            return render_template('signup.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('signup.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('signup.html')
        
        connection = get_db_connection()
        if connection:
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM login WHERE email = %s", (email,))
            if cursor.fetchone():
                flash('Email already registered (try Google login?)', 'error')
                cursor.close()
                connection.close()
                return render_template('signup.html')
            cursor.close()
            connection.close()
        
        hashed_password = generate_password_hash(password)
        
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor()
                cursor.execute("""
                    INSERT INTO login (username, email, password) 
                    VALUES (%s, %s, %s)
                """, (username, email, hashed_password))
                connection.commit()
                flash('You are successfully registered! Please login.', 'success')
                return redirect(url_for('login'))
            except Error as e:
                if "Duplicate entry" in str(e):
                    flash('Email or username already exists.', 'error')
                else:
                    print(f"Registration error: {e}")
                    flash('Registration failed. Please try again.', 'error')
            finally:
                cursor.close()
                connection.close()
        else:
            flash('Database connection failed. Please try again later.', 'error')
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login (traditional or Google)"""
    if request.method == 'POST':
        email = request.form['email'].strip()
        password = request.form['password'].strip()
        
        if not email or not password:
            flash('Please enter both email and password', 'error')
            return render_template('login.html')
        
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor()
                cursor.execute("SELECT * FROM login WHERE email = %s", (email,))
                account = cursor.fetchone()
                
                if account and check_password_hash(account[3], password):
                    session['loggedin'] = True
                    session['id'] = account[0]
                    session['username'] = account[1]
                    session['email'] = account[2]
                    flash(f'Welcome back, {account[1]}!', 'success')
                    return redirect(url_for('home'))
                else:
                    flash('Incorrect email or password!', 'error')
                    
            except Error as e:
                print(f"Login error: {e}")
                flash('Login failed due to database error. Please try again.', 'error')
            finally:
                cursor.close()
                connection.close()
        else:
            flash('Database connection failed. Please check your configuration.', 'error')
    
    return render_template('login.html')

if __name__ == '__main__':
    # Debug: Print OAuth configuration
    print("🔍 OAuth Configuration:")
    print(f"   GOOGLE_CLIENT_ID: {'Set' if os.getenv('GOOGLE_CLIENT_ID') else 'Not set'}")
    print(f"   GOOGLE_CLIENT_SECRET: {'Set' if os.getenv('GOOGLE_CLIENT_SECRET') else 'Not set'}")
    print(f"   Server Metadata URL: https://accounts.google.com/.well-known/openid-configuration")
    
    db_status = "✅ Ready" if db_success else "❌ Failed"
    dir_status = "✅ Ready" if dir_success else "❌ Failed"
    face_status = "✅ Ready" if face_detector is not None else "❌ Failed"
    google_status = "✅ Ready" if os.getenv('GOOGLE_CLIENT_ID') and os.getenv('GOOGLE_CLIENT_SECRET') else "❌ Missing credentials"
    
    print(f"\n📊 Initial status:")
    print(f"   - Database: {db_status}")
    print(f"   - Directories: {dir_status}")
    print(f"   - Face Detector: {face_status}")
    print(f"   - Google OAuth: {google_status}")
    print(f"   - Total Users: {totalreg()}")
    
    if not all([db_success, dir_success, face_detector is not None]):
        print("\n❌ Critical initialization failed. Please fix the errors above.")
        print("Common fixes:")
        print("1. Start MySQL service")
        print("2. Create 'face_detection' database")
        print("3. Download haarcascade_frontalface_default.xml")
        print("4. Check directory permissions")
        exit(1)
    
    print(f"\n🌐 Starting Flask server...")
    print(f"   Open: http://localhost:5000")
    print(f"   Press Ctrl+C to stop")
    
    app.run(debug=True, host='0.0.0.0', port=5000)