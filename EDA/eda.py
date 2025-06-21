import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

df = pd.read_csv('loan_prediction.csv')
profile = ProfileReport(df, title="Profiling Report")
ProfileReport.to_file(profile, "profiling_report.html")

print("Dataset Shape:", df.shape)
print("First 5 rows:\n", df.head())
print("\nData Types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDescriptive Statistics:\n", df.describe())

sns.countplot(x='Loan_Status', data=df)
plt.title('Loan Status Distribution')
plt.savefig('loan_status_distribution.png')
plt.clf()

sns.histplot(df['ApplicantIncome'], kde=True)
plt.title('Applicant Income Distribution')
plt.savefig('applicant_income_distribution.png')
plt.clf()


sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.clf()
