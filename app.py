import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(page_title="Breast Cancer Classifier", layout="centered")
st.title("🎗️ Breast Cancer Classification with Random Forest")

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Show raw data
if st.checkbox("Show raw dataset"):
    st.dataframe(X)

# Train-test split
test_size = st.sidebar.slider("Test set size (%)", min_value=10, max_value=50, value=20, step=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

# Random Forest parameters
n_estimators = st.sidebar.slider("Number of trees", 10, 200, 100, step=10)
max_depth = st.sidebar.slider("Max depth of trees", 1, 20, 5)

# Train model
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Model Evaluation")
st.success(f"✅ Accuracy: {accuracy:.2f}")

st.text("📋 Classification Report:")
st.text(classification_report(y_test, y_pred, target_names=data.target_names))

# Confusion matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)
