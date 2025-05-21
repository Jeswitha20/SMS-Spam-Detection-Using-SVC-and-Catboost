import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from catboost import CatBoostClassifier
import joblib
import os

try:
    data = pd.read_csv('spamdataset1.csv', encoding='latin-1', sep=',', usecols=[0, 1], names=['label', 'message'], skiprows=1)
except Exception as e:
    print("Error loading dataset:", e)
    exit()

# Drop rows with missing text or label
data.dropna(subset=['message', 'label'], inplace=True)
print(f" After dropping NaNs: {len(data)} messages")

# Convert labels to binary (ham=0, spam=1) AFTER ensuring correct formatting
data['label'] = data['label'].str.strip().map({'ham': 0, 'spam': 1})

# Drop any remaining NaN values in labels
data.dropna(subset=['label'], inplace=True)
print(f" After ensuring labels are clean: {len(data)} messages")

# Ensure dataset is clean but DO NOT remove valid messages
print(f" Dataset size before final cleanup: {len(data)} messages")

# Check if the dataset is empty after preprocessing
if data.empty:
    print("Error: No valid data left after preprocessing. Check your dataset!")
    exit()

# Split data into features and labels
X = data['message']
y = data['label']

# Convert text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
try:
    X_vectorized = vectorizer.fit_transform(X)
except ValueError as e:
    print("Error during vectorization:", e)
    exit()

# Print vocabulary size
print(f" Vocabulary size: {len(vectorizer.get_feature_names_out())} words")

# Ensure enough samples for train-test split
if len(X) < 5:
    print(" Error: Not enough valid messages for training. Check dataset!")
    exit()

# Split data into train-test
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Train SVC model with TF-IDF features and balanced class weights
svc_model = SVC(probability=True, C=2.0, kernel="linear", class_weight="balanced")
svc_model.fit(X_train, y_train)
joblib.dump(svc_model, "models/svc_model.pkl")

# Train CatBoost model
catboost_model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=0)
catboost_model.fit(X_train, y_train)
joblib.dump(catboost_model, "models/catboost_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "models/vectorizer.pkl")

print(" Models trained with TF-IDF and balanced class weights, and saved successfully!")

from sklearn.metrics import accuracy_score

# Train SVC model with TF-IDF features and balanced class weights
svc_model = SVC(probability=True, C=2.0, kernel="linear", class_weight="balanced")
svc_model.fit(X_train, y_train)

# Predictions
svc_train_preds = svc_model.predict(X_train)
svc_test_preds = svc_model.predict(X_test)

# Calculate accuracies
svc_train_acc = accuracy_score(y_train, svc_train_preds)
svc_test_acc = accuracy_score(y_test, svc_test_preds)

print(f"SVC Train Accuracy: {svc_train_acc:.4f}")
print(f"SVC Test Accuracy: {svc_test_acc:.4f}")

# Save the model
joblib.dump(svc_model, "models/svc_model.pkl")

# Train CatBoost model
catboost_model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=0)
catboost_model.fit(X_train, y_train)

# Predictions
catboost_train_preds = catboost_model.predict(X_train)
catboost_test_preds = catboost_model.predict(X_test)

# Calculate accuracies
catboost_train_acc = accuracy_score(y_train, catboost_train_preds)
catboost_test_acc = accuracy_score(y_test, catboost_test_preds)

print(f"CatBoost Train Accuracy: {catboost_train_acc:.4f}")
print(f"CatBoost Test Accuracy: {catboost_test_acc:.4f}")

# Save the model
joblib.dump(catboost_model, "models/catboost_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "models/vectorizer.pkl")

print(" Models trained with TF-IDF and balanced class weights, and saved successfully!")
print("Training completed successfully!")