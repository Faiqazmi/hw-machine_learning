# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the data
data = pd.read_csv('https://raw.githubusercontent.com/Faiqazmi/Dataset_latihan/main/Prediction%20Insurance.csv')

# Exploratory Data Analysis
print(data.head())
print(data.info())
print(data.describe())

# Preprocessing Data
# Encoding categorical variables
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['Vehicle_Age'] = data['Vehicle_Age'].map({'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2})
data['Vehicle_Damage'] = data['Vehicle_Damage'].map({'No': 0, 'Yes': 1})

# Features and target variable
X = data.drop(columns=['id', 'Response'])
y = data['Response']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model
with open('insurance_model.pkl', 'wb') as file:
    pickle.dump(model, file)
