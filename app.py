import os
import numpy as np
import pandas as pd
from datetime import date, datetime
from flask import Flask, request, render_template, redirect, url_for, session, flash
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length
import cv2
import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from authlib.integrations.flask_client import OAuth
import requests
from sklearn.neighbors import KNeighborsClassifier
import joblib
import base64
import io
from PIL import Image
import boto3
from botocore.exceptions import ClientError

# Load environment variables
load_dotenv()

# Flask App
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'fallback_secret_key')

# AWS S3 Configuration
S3_BUCKET = os.getenv('S3_BUCKET', 'your-bucket-name')
S3_REGION = os.getenv('S3_REGION', 'us-east-1')
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=S3_REGION
)

# Flask-WTF Form for Add User
class AddUserForm(FlaskForm):
    newusername = StringField('User Name', validators=[DataRequired(), Length(min=3, max=100)])
    newuserid = StringField('User ID', validators=[DataRequired(), Length(min=1, max=20)])
    submit = SubmitField('Add New User')

# Google OAuth2 Configuration
app.config['OAUTH2_PROVIDERS'] = {
    'google': {
        'client_id': os.getenv('GOOGLE_CLIENT_ID'),
        'client_secret': os.getenv('GOOGLE_CLIENT_SECRET'),
        'authorize_url': 'https://accounts.google.com/o/oauth2/v2/auth',
        'access_token_url': 'https://oauth2.googleapis.com/token',
        'api_base_url': 'https://www.googleapis.com/oauth2/v2/',
        'client_kwargs': {'scope': 'openid email profile'},
        'userinfo_endpoint': 'https://openidconnect.googleapis.com/v1/userinfo',
        'server_metadata_url': 'https://accounts.google.com/.well-known/openid-configuration'
    }
}

# Register Google provider
oauth = OAuth(app)
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
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# MySQL Configuration
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST', 'localhost')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', 'your_password')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB', 'face_detection')

def get_db_connection():
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
    print("=== Initializing directories ===")
    dirs_to_create = ['/tmp/Attendance', '/tmp/static', '/tmp/static/faces']
    success = True
    for dir_path in dirs_to_create:
        if not safe_makedirs(dir_path):
            success = False
    csv_path = f'/tmp/Attendance/Attendance-{datetoday}.csv'
    if not os.path.exists(csv_path):
        try:
            with open(csv_path, 'w') as f:
                f.write('Name,Roll,Time\n')
            s3_client.upload_file(csv_path, S3_BUCKET, f'Attendance/Attendance-{datetoday}.csv')
            print(f"✅ Created and uploaded attendance CSV: {csv_path}")
        except Exception as e:
            print(f"❌ Error creating CSV: {e}")
            success = False
    else:
        print(f"✅ Attendance CSV exists: {csv_path}")
    print(f"Directory setup: {'✅ SUCCESS' if success else '❌ FAILED'}")
    return success

def init_db():
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
    try:
        cascade_path = 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            print("❌ ERROR: haarcascade_frontalface_default.xml not found!")
            return None
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            print("❌ ERROR: Failed to load haarcascade file!")
            return None
        print("✅ Face detector initialized successfully")
        return detector
    except Exception as e:
        print(f"❌ Error initializing face detector: {e}")
        return None

# Initialize at startup
print("🚀 Starting Face Recognition Attendance System...")
print(f"📅 Today's date: {datetoday2}")
db_success = init_db()
dir_success = init_directories()
face_detector = init_face_detector()

def totalreg():
    try:
        faces_dir = '/tmp/static/faces'
        if not os.path.exists(faces_dir) or not os.path.isdir(faces_dir):
            return 0
        return len([d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))])
    except:
        return 0

def extract_faces(img):
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
    try:
        model_path = '/tmp/static/face_recognition_model.pkl'
        if not os.path.exists(model_path):
            try:
                s3_client.download_file(S3_BUCKET, 'face_recognition_model.pkl', model_path)
            except ClientError:
                print("⚠️  No trained model found in S3. Please add users first.")
                return None
        model = joblib.load(model_path)
        prediction = model.predict(facearray)[0]
        return prediction
    except Exception as e:
        print(f"Face recognition error: {e}")
        return None

def train_model():
    try:
        faces_dir = '/tmp/static/faces'
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
                    # Upload to S3
                    s3_key = f'faces/{user}/{imgname}'
                    s3_client.upload_file(img_path, S3_BUCKET, s3_key)
                    print(f"✅ Uploaded {s3_key} to S3")
        if len(faces) > 0:
            faces = np.array(faces)
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(faces, labels)
            model_path = '/tmp/static/face_recognition_model.pkl'
            joblib.dump(knn, model_path)
            s3_client.upload_file(model_path, S3_BUCKET, 'face_recognition_model.pkl')
            print(f"✅ Model trained with {total_images} face samples from {len(userlist)} users and uploaded to S3")
            return True
        else:
            print("⚠️  No face data available for training")
            return False
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False

def extract_attendance():
    try:
        csv_path = f'/tmp/Attendance/Attendance-{datetoday}.csv'
        s3_key = f'Attendance/Attendance-{datetoday}.csv'
        try:
            s3_client.download_file(S3_BUCKET, s3_key, csv_path)
        except ClientError:
            with open(csv_path, 'w') as f:
                f.write('Name,Roll,Time\n')
            s3_client.upload_file(csv_path, S3_BUCKET, s3_key)
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
    try:
        username = name.split('_')[0]
        userid = name.split('_')[1]
        current_time = datetime.now().strftime("%H:%M:%S")
        csv_path = f'/tmp/Attendance/Attendance-{datetoday}.csv'
        s3_key = f'Attendance/Attendance-{datetoday}.csv'
        df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(columns=['Name', 'Roll', 'Time'])
        if int(userid) not in list(df['Roll']):
            new_record = pd.DataFrame({
                'Name': [name],
                'Roll': [userid],
                'Time': [current_time]
            })
            df = pd.concat([df, new_record], ignore_index=True)
            df.to_csv(csv_path, index=False)
            s3_client.upload_file(csv_path, S3_BUCKET, s3_key)
            print(f"✅ Uploaded attendance CSV to S3: {s3_key}")
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

@app.route('/start', methods=['GET', 'POST'])
def start():
    if 'loggedin' not in session:
        flash('Please login to access this page', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            image_file = request.files['image']
            image_data = image_file.read()
            image = Image.open(io.BytesIO(image_data))
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            faces = extract_faces(image)
            if not faces:
                flash('No face detected in the uploaded image.', 'error')
                return redirect(url_for('start'))
            (x, y, w, h) = faces[0]
            face = cv2.resize(image[y:y+h, x:x+w], (50, 50))
            face_array = face.reshape(1, -1)
            identified_person = identify_face(face_array)
            if identified_person:
                if add_attendance(identified_person):
                    flash(f'Marked as present for {identified_person}', 'success')
                else:
                    flash(f'Attendance already marked for {identified_person}', 'warning')
            else:
                flash('Unknown face detected.', 'error')
            return redirect(url_for('home'))
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            flash('Error processing image. Please try again.', 'error')
            return redirect(url_for('start'))
    
    return render_template('start.html')

@app.route('/add', methods=['GET', 'POST'])
def add():
    if 'loggedin' not in session:
        flash('Please login to access this page', 'warning')
        return redirect(url_for('login'))
    
    form = AddUserForm()
    if form.validate_on_submit():
        newusername = form.newusername.data.strip()
        newuserid = form.newuserid.data.strip()
        userimagefolder = f'/tmp/static/faces/{newusername}_{newuserid}'
        try:
            if not os.path.isdir(userimagefolder):
                os.makedirs(userimagefolder)
            images = request.files.getlist('images')
            if len(images) < nimgs:
                flash(f'Please upload exactly {nimgs} images.', 'error')
                return render_template('add.html', form=form, totalreg=totalreg(), datetoday2=datetoday2)
            for i, image_file in enumerate(images[:nimgs]):
                image_data = image_file.read()
                image = Image.open(io.BytesIO(image_data))
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                faces = extract_faces(image)
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    name = f'{newusername}_{i}.jpg'
                    img_path = os.path.join(userimagefolder, name)
                    cv2.imwrite(img_path, image[y:y+h, x:x+w])
                    s3_key = f'faces/{newusername}_{newuserid}/{name}'
                    s3_client.upload_file(img_path, S3_BUCKET, s3_key)
                    print(f"✅ Uploaded {s3_key} to S3")
                else:
                    flash(f'No face detected in image {i+1}.', 'error')
                    return render_template('add.html', form=form, totalreg=totalreg(), datetoday2=datetoday2)
            if train_model():
                flash(f'User {newusername} added and model trained successfully!', 'success')
            else:
                flash(f'User {newusername} added but model training failed.', 'warning')
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
            names, rolls, times, l = extract_attendance()
            total_users = totalreg()
            return render_template('home.html',
                                 names=names,
                                 rolls=rolls,
                                 times=times,
                                 l=l,
                                 totalreg=total_users,
                                 datetoday2=datetoday2)
        except Exception as e:
            print(f"❌ Error adding user: {e}")
            flash('Error adding user. Please try again.', 'error')
    return render_template('add.html', form=form, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/logout')
def logout():
    if 'username' in session:
        flash(f'Goodbye, {session["username"]}!', 'info')
    session.clear()
    return redirect(url_for('login'))

@app.route('/auth/google')
def google_login():
    if not os.getenv('GOOGLE_CLIENT_ID') or not os.getenv('GOOGLE_CLIENT_SECRET'):
        flash('Google authentication not configured. Please use email login.', 'error')
        return redirect(url_for('login'))
    try:
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
    import os
    if os.getenv('FLASK_ENV') == 'production':
        from gunicorn.app.base import BaseApplication
        class StandaloneApplication(BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()
            def load_config(self):
                for key, value in self.options.items():
                    self.cfg.set(key.lower(), value)
            def load(self):
                return self.application
        options = {
            'bind': '0.0.0.0:5000',
            'workers': 1,
            'timeout': 120
        }
        StandaloneApplication(app, options).run()
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)
