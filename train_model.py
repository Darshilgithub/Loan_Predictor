# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load dataset (you can download from Kaggle - "Loan Prediction Dataset")
# For this example, I'll show how to create a sample dataset
# In practice, use: df = pd.read_csv('loan_data.csv')

# Creating sample data (replace this with actual dataset loading)
np.random.seed(42)
n_samples = 1000

data = {
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Married': np.random.choice(['Yes', 'No'], n_samples),
    'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples),
    'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
    'Self_Employed': np.random.choice(['Yes', 'No'], n_samples),
    'ApplicantIncome': np.random.randint(1000, 10000, n_samples),
    'CoapplicantIncome': np.random.randint(0, 5000, n_samples),
    'LoanAmount': np.random.randint(50, 500, n_samples),
    'Loan_Amount_Term': np.random.choice([360, 180, 120], n_samples),
    'Credit_History': np.random.choice([0.0, 1.0], n_samples, p=[0.2, 0.8]),
    'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples),
}

df = pd.DataFrame(data)

# Create target variable (Loan_Status) based on logic
df['Loan_Status'] = 'N'
df.loc[(df['Credit_History'] == 1.0) & 
       (df['ApplicantIncome'] + df['CoapplicantIncome'] > 5000) & 
       (df['Education'] == 'Graduate'), 'Loan_Status'] = 'Y'

print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values (if any)
df = df.fillna(df.mode().iloc[0])

# Encode categorical variables
le = LabelEncoder()
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                   'Self_Employed', 'Property_Area', 'Loan_Status']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("\nAfter Encoding:")
print(df.head())

# Separate features and target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

best_model = None
best_score = 0
best_name = ""

print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    if accuracy > best_score:
        best_score = accuracy
        best_model = model
        best_name = name

print("\n" + "="*50)
print(f"Best Model: {best_name} with accuracy: {best_score:.4f}")
print("="*50)

# Save the best model and scaler
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save label encoders for categorical variables
feature_names = X.columns.tolist()
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print("\nModel saved successfully!")
print("Files created: loan_model.pkl, scaler.pkl, feature_names.pkl")