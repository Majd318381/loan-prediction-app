from flask import Flask, request, jsonify, render_template
import sqlite3

app = Flask(__name__)
DB_FILE = 'loan_data.db'

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

# Create (Add new loan)
@app.route('/loans', methods=['POST'])
def add_loan():
    data = request.json
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO loans (
                gender, married, dependents, education, self_employed,
                applicant_income, coapplicant_income, loan_amount,
                loan_amount_term, credit_history, property_area
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            data.get('property_area')
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

# Update (Approve loan by ID)
@app.route('/loans/<int:loan_id>/approve', methods=['PUT'])
def approve_loan(loan_id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE loans SET status = 1 WHERE loan_id = ?', (loan_id,))
        conn.commit()
        if cursor.rowcount == 0:
            return jsonify({'error': 'Loan not found'}), 404
    return jsonify({'message': 'Loan approved'})

# Update (Reject loan by ID)
@app.route('/loans/<int:loan_id>/reject', methods=['PUT'])
def reject_loan(loan_id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE loans SET status = 0 WHERE loan_id = ?', (loan_id,))
        conn.commit()
        if cursor.rowcount == 0:
            return jsonify({'error': 'Loan not found'}), 404
    return jsonify({'message': 'Loan rejected'})

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