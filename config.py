import mysql.connector

def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='Ashok3182004.',
        database='flask_auth_db'
    )
