# Loan Prediction Project
This project is a homework assignment for the *Machine Learning Techniques (MLT)* course at the *Syrian Virtual University (SVU)*.  
**Submission deadline:** Sunday, 22 June 2025, 11:59 PM.

## Requirements
1.	Build a web application to help banks approve/reject individual requests.
2.	Add/remove a request and its output. 
3.	The web application should ___contain___ Exploratory Data Analysis over all requests
4.	Solving the ___data___ issues should be in the report/additional page in the website
5.	A page for model accuracy (recall, precision, F1, and accuracy)

## Dataset Description

The dataset provided for this project contains information about loan applicants, including demographic details, financial status, and loan application outcomes. The dataset columns are:

- **Loan_ID**: Unique identifier for each loan application
- **Gender**: Applicant's gender
- **Married**: Marital status
- **Dependents**: Number of dependents
- **Education**: Education level of the applicant
- **Self_Employed**: Employment status
- **ApplicantIncome**: Income of the applicant
- **CoapplicantIncome**: Income of the co-applicant (if any)
- **LoanAmount**: Amount of loan applied for
- **Loan_Amount_Term**: Term of the loan in months
- **Credit_History**: Credit history (1: Good, 0: Bad / No Prior Credit Record)
- **Property_Area**: Urban, Semiurban, or Rural
- **Loan_Status**: Loan approval status (Y: Approved / N: Rejected)

The dataset should be used to predict whether a loan will be approved based on applicant information.

## Solution Overview

To address the project requirements, we have:

- Implemented a lightweight database using SQLite to store loan applications.
- Developed a RESTful backend using Flask to handle application logic and data management.
- Created a frontend interface with Bootstrap for styling and jQuery for interacting with backend services.

## Backend Services
The Flask backend provides the following REST API endpoints:

- **POST `/loans`**: Submit a new loan application ___and predict the status___.
- **GET `/loans`**: Retrieve all loan applications.
- **GET `/loan/<loan_id>`**: Get details of a specific loan application by its Loan_ID.
- **DELETE `/loan/<loan_id>`**: Delete a specific loan application by its Loan_ID.

All endpoints return JSON responses for seamless frontend integration.

## Frontend
- **index.html**: Displays a list of all loan applications stored in the database.
- **create.html**: Provides a form for submitting a new loan application.
- **accuracy.html**: This page show accuracy ,precision , f1_score , recall.
- **Dataissue.html**: Show  some of the problem in dataset  and how we  soliving it .
- **expolety.html**: This page explain the data and some features in data and relationship between columns.

## Model Training
- Reads the loan prediction dataset from a CSV file.
- Drops the Loan_ID column.
- Fills missing values in all columns using the most frequent value.
- Encodes categorical columns into numeric values using LabelEncoder and saves the encoders for future use.
- Separates the features (X) from the target variable (y, which is Loan_Status).
- Splits the data into training and testing sets.
- Tries logistic regression (commented out due to convergence issues).
- Trains a RandomForestClassifier on the training data.
- Predicts on the test set and prints accuracy, precision, recall, and F1 score.
- Saves the trained model and the encoders to disk for later use.

## Hosting
We will use www.pythonanywhere.com free account

create a new web app
select web framework Flask
select python version 3.10
Update WSGI configuration 

After creating the account we will setup the venv on bash terminal and install the libraries.
  mkvirtualenv venv --python=python3.10
  pip install flask

https://yusuf233336.pythonanywhere.com/

## Dependencies

All required Python libraries for this project are listed in `requirements.txt`.  
To install them, run:

```bash
pip install -r requirements.txt
```

## Exploratory Data Analysis (EDA):
- We used ydata-prfiling library to generate a complete data profiling report.
- You can reach the report from the top menu in the app.
- EDA/eda.py contains the script to generate the report and some additional charts:
  - Loan Status Distribution
  - Applicant Income Distribution
  - Correlation Heatmap

## Model Training:
- Location: Model_Training/train_model.py
- Logistic Regression model trained and saved as 'loan_model.pkl'
- Evaluation metrics printed to console.

## Date Issue:
No Date or Time fields found in dataset. No handling needed.

___I guess it was a typo, he meant to say data issues not date.___

## TODO
- review the model training and data preprocessing
- Add a Model Accuracy page: Provide a page that presents model evaluation metrics such as recall, precision, F1 score, and accuracy.
