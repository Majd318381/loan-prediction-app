import sqlite3

# Connect to SQLite database (creates file if it doesn't exist)
conn = sqlite3.connect('loan_data.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS loans (
        loan_id INTEGER PRIMARY KEY AUTOINCREMENT,
        gender VARCHAR(10),
        married VARCHAR(5),
        dependents VARCHAR(5),
        education VARCHAR(15),
        self_employed VARCHAR(5),
        applicant_income FLOAT,
        coapplicant_income FLOAT,
        loan_amount FLOAT,
        loan_amount_term INTEGER,
        credit_history INTEGER,
        property_area VARCHAR(15),
        loan_status CHAR(1)
    )
''')

conn.commit()