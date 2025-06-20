import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('loan_prediction.csv')

# Preprocessing the dataset
df = df.drop('Loan_ID', axis=1)

# Fill missing values for categorical columns with the most frequent value
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

df.info()

# Convert categorical columns to numerical using Label Encoding
encoders = {}
for column in df.select_dtypes(include='object').columns.tolist():
    print(f"Encoding column: {column}")
    le = LabelEncoder()
    df_imputed[column] = le.fit_transform(df_imputed[column])
    encoders[column] = le

# Save encoders for future use
joblib.dump(encoders, 'encoders.pkl')
print("Encoders are saved as encoders.pkl")

# X is the feature set, y is the target variable
X = df_imputed.drop('Loan_Status', axis=1)
y = df_imputed['Loan_Status']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
# model = LogisticRegression(max_iter=2000)
# model.fit(X_train, y_train)
# Stopped using this model due to convergence issues, see the warning below
## ConvergenceWarning: lbfgs failed to converge after 2000 iteration(s) (status=1):
## STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluating the model
y_pred = model.predict(X_test)

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Saving the model
joblib.dump(model, 'loan_model.pkl')
print("Model saved as loan_model.pkl")
