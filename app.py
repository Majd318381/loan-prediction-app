from flask import Flask, request, jsonify, render_template
import sqlite3
import joblib
import numpy as np

app = Flask(__name__)
DB_FILE = 'loan_data.db'

# Load the saved model
model = joblib.load('Model_Training/loan_model.pkl')
encoders = joblib.load('Model_Training/encoders.pkl')

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/create')
def create():
    return render_template('create.html')

@app.route('/profiling-report')
def profiling_report():
    try:
        with open('EDA/profiling_report.html', 'r', encoding='utf-8') as f:
            report_html = f.read()
        return report_html
    except FileNotFoundError:
        return "Profiling report not found.", 404
    
# Create (Add new loan)
@app.route('/loans', methods=['POST'])
def add_loan():
    data = request.json
    print("Received data:", data)

    # Convert categorical features to the format expected by the model if needed
    encoded_data = {}
    for column, le in encoders.items():
        column = column.lower()
        if column == 'loan_status':
            continue
        print(f"Column: {column}, Before encoding: {data.get(column)}")
        encoded_value = le.transform([data.get(column)])[0]
        print(f"Column: {column}, After encoding: {encoded_value}")
        encoded_data[column] = encoded_value
    
    # Prepare input features for prediction

    features = [
        encoded_data.get('gender'),
        encoded_data.get('married'),
        encoded_data.get('dependents'),
        encoded_data.get('education'),
        encoded_data.get('self_employed'),
        data.get('applicant_income'),
        data.get('coapplicant_income'),
        data.get('loan_amount'),
        data.get('loan_amount_term'),
        data.get('credit_history'),
        encoded_data.get('property_area')
    ]

    # Convert features to numpy array and reshape for prediction
    input_array = np.array(features).reshape(1, -1)

    # Predict the loan status
    predicted_status = int(model.predict(input_array)[0])

    # Decode the predicted_status using the loan_status encoder
    loan_status_label = encoders['Loan_Status'].inverse_transform([predicted_status])[0]
    data['loan_status'] = loan_status_label

    # Save the loan data to the database
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO loans (
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_amount_term, credit_history, property_area, loan_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('gender'),
            data.get('married'),
            data.get('dependents'),
            data.get('education'),
            data.get('self_employed'),
            data.get('applicant_income'),
            data.get('coapplicant_income'),
            data.get('loan_amount'),
            data.get('loan_amount_term'),
            data.get('credit_history'),
            data.get('property_area'),
            data.get('loan_status')
        ))
        conn.commit()
        loan_id = cursor.lastrowid
    return jsonify({'loan_id': loan_id}), 201

# Read (Get all loans)
@app.route('/loans', methods=['GET'])
def get_loans():
    with get_db_connection() as conn:
        loans = conn.execute('SELECT * FROM loans').fetchall()
        return jsonify([dict(row) for row in loans])

# Read (Get single loan by ID)
@app.route('/loans/<int:loan_id>', methods=['GET'])
def get_loan(loan_id):
    with get_db_connection() as conn:
        loan = conn.execute('SELECT * FROM loans WHERE loan_id = ?', (loan_id,)).fetchone()
        if loan is None:
            return jsonify({'error': 'Loan not found'}), 404
        return jsonify(dict(loan))

# Delete (Remove loan by ID)
@app.route('/loans/<int:loan_id>', methods=['DELETE'])
def delete_loan(loan_id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM loans WHERE loan_id = ?', (loan_id,))
        conn.commit()
        if cursor.rowcount == 0:
            return jsonify({'error': 'Loan not found'}), 404
    return jsonify({'message': 'Loan deleted'})



if __name__ == '__main__':
    app.run(debug=True)