import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('Fashion_Retail_Sales.csv')

# Drop rows with missing Purchase Amount
df = df.dropna(subset=['Purchase Amount (USD)'])

# Convert date to datetime and extract useful features
df['Date Purchase'] = pd.to_datetime(df['Date Purchase'], dayfirst=True, errors='coerce')
df['Purchase Month'] = df['Date Purchase'].dt.month
df['Purchase Day'] = df['Date Purchase'].dt.day
df['Purchase Weekday'] = df['Date Purchase'].dt.weekday

# Drop unneeded columns
df = df.drop(columns=['Customer Reference ID', 'Date Purchase'])

# Define features and target
X = df.drop('Purchase Amount (USD)', axis=1)
y = df['Purchase Amount (USD)']

# Preprocessing pipeline
categorical_features = ['Item Purchased', 'Payment Method']
numerical_features = ['Review Rating', 'Purchase Month', 'Purchase Day', 'Purchase Weekday']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Full pipeline with model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model_pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(model_pipeline, 'purchase_amount_model.pkl')

print("Model training completed and saved as 'purchase_amount_model.pkl'.")
