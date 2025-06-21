import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Read Data and Replacing null value
data = pd.read_csv('loan_prediction.csv') #import the train dataset

data = data.drop(['Loan_ID'], axis = 1)

data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])

# Fill missing values in numerical column
# We can fill in the missing values of the loan amount column with the median value.
# The median is an appropriate measure to fill in missing values when dealing with 
# skewed distributions or when outliers are present in the data.
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())

# We can fill in the missing values of the loan amount term column with the mode value of the column. 
# Since the term of the loan amount is a discrete value, the mode is an appropriate metric to use
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])


# We can fill in the missing values of the credit history column with the mode value.
# Since credit history is a binary variable (0 or 1), the mode represents the most common value 
# and is an appropriate choice for filling in missing values.
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])

# Removing outliers from data by IQR

# The “ApplicantIncome” column contains outliers which need to be removed before moving further. 
# Here’s how to remove the outliers:

# Calculate the IQR
Q1 = data['ApplicantIncome'].quantile(0.25)
Q3 = data['ApplicantIncome'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
data = data[(data['ApplicantIncome'] >= lower_bound) & (data['ApplicantIncome'] <= upper_bound)]

# The income of the loan co-applicant also contains outliers. Let’s remove the outliers from this column as well:

# Calculate the IQR
Q1= data['CoapplicantIncome'].quantile(0.25)
Q3 = data['CoapplicantIncome'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
data= data[(data['CoapplicantIncome'] >= lower_bound) & (data['CoapplicantIncome'] <= upper_bound)]

# Scale the numerical columns using StandardScaler
scaler = StandardScaler()
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
data[numerical_cols] = scaler.transform(data[numerical_cols])

# Convert categorical Columns into numerical ones
encoders = {}
col_to_encode = ['Gender','Dependents', 'Married', 'Education', 'Self_Employed', 'Property_Area','Loan_Status']
for i in col_to_encode:
    le = LabelEncoder()
    data[i] = le.fit_transform(data[i])
    encoders[i] = le

# Save encoders for future use
joblib.dump(encoders, 'encoders.pkl')

#  Split the dataset into features (X) and target (y)
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

# Fit model 
model = LogisticRegression() #define the model
model.fit(X_train, y_train) #fit the model
y_pred = model.predict(X_test) #predict on test sample

# print accuaracy ,precision ,recall , and f1_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Save the trained model to a file
joblib.dump(model, 'loan_model.pkl')