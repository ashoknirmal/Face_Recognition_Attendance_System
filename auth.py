
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, session, g
from werkzeug.security import generate_password_hash, check_password_hash
from config import get_db_connection

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Use the secret key from environment variables
app.secret_key = os.getenv('SECRET_KEY')

# Before request: Set up the database connection for each request
@app.before_request
def before_request():
    g.db = get_db_connection()
    g.cursor = g.db.cursor()

# Teardown: Close the database connection after each request
@app.teardown_request
def teardown_request(exception):
    g.db.close()

# Sign-Up Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Encrypt the password
        hashed_password = generate_password_hash(password, method='sha256')
        
        # Insert user into the database
        g.cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", 
                         (username, email, hashed_password))
        g.db.commit()
        
        return redirect('/login')
    return render_template('signup.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Retrieve user from database
        g.cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = g.cursor.fetchone()
        
        # Check if user exists and password matches
        if user and check_password_hash(user[3], password):
            session['user'] = user[1]  # Store the username in session
            return redirect('/')
        else:
            return render_template('login.html', message="Invalid email or password.")
    return render_template('login.html')

# # Home Route
# @app.route('/')
# def home():
#     if 'user' in session:
#         return f"Hello, {session['user']}!"
#     return redirect('/login')


@app.route('/')
def home():
    if 'user' in session:
        # Pass a welcome message and the username to the home page
        return render_template('home.html', username=session['user'], message="Welcome to Face Detection")
    else:
        return redirect('/login')
    

# Logout Route
@app.route('/logout')
def logout():
    session.pop('user', None)  # Clear session
    return redirect('/login')

# Favicon Route (to handle favicon request and avoid 404 error)
@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)
