# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 18:37:48 2024

@author: reddy
"""

import streamlit as st
import os
from PIL import Image
from ultralytics import YOLO
import yaml

# Title and Description
st.title("Skin Issue Detection using YOLOv8")
st.markdown("This app detects skin diseases using a YOLOv8 model trained on custom data.")

# Upload an Image
uploaded_file = st.file_uploader("Upload an image for detection", type=["jpg", "png", "jpeg"])

# Path to your YOLOv8 model weights and data.yaml file
model_path = r"C:\Users\reddy\AI\models\yolov8n.pt"  # Update the path to your YOLOv8 model weights
data_yaml_path = r"C:\Users\reddy\AI\dataset\data.yaml"  # Update this to the path of your data.yaml

# Load class names from the data.yaml file
def load_class_names(data_yaml_path):
    with open(data_yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']  # Extract class names from the YAML

class_names = load_class_names(data_yaml_path)

if uploaded_file is not None:
    # Save uploaded file locally
    input_image_path = os.path.join("uploaded_image.jpg")
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(Image.open(uploaded_file), caption="Uploaded Image", use_column_width=True)

    # Run YOLOv8 Detection
    st.write("Running YOLOv8 model for detection...")

    try:
        # Load YOLOv8 model
        model = YOLO(model_path)

        # Perform inference on the uploaded image
        results = model(input_image_path)

        # Get results for the first detection
        result = results[0]

        # Check if there are detections
        if len(result.boxes) == 0:
            st.write("No skin issues detected in the image.")
        else:
            # Plot the results (this will draw bounding boxes)
            results_img = result.plot()  # This will plot the bounding boxes
            result_img = Image.fromarray(results_img)
            st.image(result_img, caption="Detection Result", use_column_width=True)

            # Display detected labels and confidence
            st.write("Detection Results:")
            for i in range(len(result.boxes)):  # Loop through detected boxes
                # Get the class ID from the detection
                class_id = int(result.boxes.cls[i].item())  # Ensure class ID is an integer
                class_name = class_names[class_id]  # Get the class name from the data.yaml
                confidence = result.boxes.conf[i].item() * 100  # Convert confidence to percentage
                
                # Make sure class name is correct and display the information
                st.write(f"Class: {class_name}, Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error(f"Error during detection: {str(e)}")

# Footer
st.markdown("Powered by YOLOv8 and Streamlit. Built by Kiran reddy.")
