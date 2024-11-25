import streamlit as st
import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained KNN model and LabelEncoder
knn_crop = joblib.load('knn_crop_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Streamlit UI
st.title("Crop Recommendation System Based on Soil Quality")

# User input for soil parameters using sliders and text input
ph = st.slider('pH of Soil', min_value=0.0, max_value=14.0, value=6.5, step=0.1)
N = st.slider('Nitrogen Content (N)', min_value=0.0, max_value=100.0, value=50.0, step=1.0)
P = st.slider('Phosphorus Content (P)', min_value=0.0, max_value=100.0, value=30.0, step=1.0)
K = st.slider('Potassium Content (K)', min_value=0.0, max_value=100.0, value=20.0, step=1.0)
temperature = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, value=28.5, step=0.1)
humidity = st.slider('Humidity (%)', min_value=0.0, max_value=100.0, value=60.0, step=1.0)
rainfall = st.slider('Rainfall (mm)', min_value=0.0, max_value=2000.0, value=100.0, step=1.0)

# Button to predict the crop
if st.button("Predict Crop"):
    # Prepare input data for prediction
    test_input = pd.DataFrame([[ph, N, P, K, temperature, humidity, rainfall]],
                              columns=['ph', 'N', 'P', 'K', 'temperature', 'humidity', 'rainfall'])

    # Make predictions using the KNN model
    predicted_crop = knn_crop.predict(test_input)

    # Decode the predicted crop label
    predicted_crop_name = label_encoder.inverse_transform(predicted_crop)[0]

    # Display the recommended crop
    st.write(f"Recommended Crop: {predicted_crop_name}")
    
    # Load test data (for accuracy evaluation)
    # Replace with your actual test dataset path
    data = pd.read_csv('soil_quality_dataset.csv')
    X_test = data[['ph', 'N', 'P', 'K', 'temperature', 'humidity', 'rainfall']]
    y_test = data['label']
    
    # Predict on test set and calculate accuracy
    y_test_encoded = label_encoder.transform(y_test)  # Encode test labels
    y_pred = knn_crop.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    
    # Display the accuracy
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    st.pyplot(plt)
    
    # Classification Report
    class_report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_)
    st.text_area("Classification Report", class_report, height=200)
    
    # Crop Distribution Graph (Bar Chart)
    crop_counts = data['label'].value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=crop_counts.index, y=crop_counts.values, palette="viridis")
    plt.title("Distribution of Crops in the Dataset")
    plt.xlabel("Crop Type")
    plt.ylabel("Count")
    st.pyplot(plt)
    
    # Plot for predicted crop
    plt.figure(figsize=(8, 5))
    sns.barplot(x=[predicted_crop_name], y=[1], palette="Blues")
    plt.title(f"Predicted Crop: {predicted_crop_name}")
    plt.xlabel("Crop")
    plt.ylabel("Prediction Confidence")
    st.pyplot(plt)
