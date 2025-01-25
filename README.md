# Skin-issue-detection
This project is designed to detect and classify various skin issues using computer vision techniques. The model leverages the **YOLOv8** (Ultralytics YOLOv8) deep learning model for object detection and image classification. The goal of this project is to help in the early detection and diagnosis of skin conditions by analyzing images and identifying potential issues. The interface is built using **Streamlit**, making it user-friendly and interactive.

## Project Features
- **Skin Issue Detection**: Classify various skin conditions like acne, eczema, psoriasis, etc.
- **YOLOv8 Model**: Utilizes the **Ultralytics YOLOv8** model for image classification and real-time detection.
- **Real-time Image Analysis**: Capable of processing and detecting skin issues in real-time images.
- **Streamlit Interface**: A web app interface for easy image upload and real-time detection.

## Technologies Used
- **Python**: Programming language used for development.
- **Ultralytics YOLOv8**: Object detection model for classifying skin issues.
- **OpenCV**: Image processing library for real-time video feed.
- **PIL (Python Imaging Library)**: For image preprocessing and manipulation.
- **Streamlit**: Framework for building the interactive web app.
- **YAML**: Configuration file handling and model settings.

## Installation Instructions

1. Clone this repository to your local machine:
    ```
    git clone https://github.com/your-username/skin-issue-detection.git
    ```

2. Navigate into the project directory:
    ```
    cd skin-issue-detection
    ```

3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

4. Create a configuration file (`config.yaml`) with your model parameters:
    Example `config.yaml`:
    ```yaml
    model: yolov8
    image_size: 640
    confidence_threshold: 0.5
    ```

## Usage

1. **Prepare your dataset**: Ensure you have labeled skin images for training or testing.
2. **Train the model**: You can train the YOLOv8 model using the `train_model.py` script.
    ```
    python train_model.py
    ```
    This script will use your dataset to train the YOLOv8 model to recognize different skin issues.

3. **Run the Streamlit app**: After training the model, you can use the Streamlit app to upload images and detect skin issues.
    ```
    streamlit run app.py
    ```
    This will launch the Streamlit interface in your web browser where you can upload images for detection.

4. **Detect skin issues**: Once the app is running, upload an image through the interface, and the model will detect and classify the skin issues based on the uploaded image.

5. **Real-time detection**: You can also use the app for real-time skin issue detection via your webcam. The model will process the live video feed and detect any skin issues in real-time.


