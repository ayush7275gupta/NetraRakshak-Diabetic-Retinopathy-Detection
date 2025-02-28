# Netra Rakshak - Diabetic Retinopathy Detection

## Introduction
Netra Rakshak is a cutting-edge machine learning project designed to detect diabetic retinopathy in retinal images. Diabetic retinopathy is a severe complication of diabetes that can lead to permanent vision loss if not identified and treated early. Leveraging the power of deep learning, Netra Rakshak aims to assist medical professionals in the timely detection of this condition, thereby improving patient outcomes and reducing the risk of blindness.

## Why Netra Rakshak?
Diabetic retinopathy is one of the leading causes of preventable blindness worldwide. Early detection is critical, but traditional methods of retinal screening can be time-consuming and require significant expertise. Netra Rakshak addresses these challenges by providing:
  1. Automated Detection: Reduces the time needed for manual screening.
  2. High Accuracy: Utilizes state-of-the-art deep learning models for reliable predictions.
  3. Scalability: Can be deployed in both clinical and remote settings, enabling broader access to healthcare.
  4. Cost Efficiency: Offers a low-cost solution compared to traditional screening methods.

## Features
  1. Deep Learning-Based Detection: Classifies images into severity levels of diabetic retinopathy
  2. Early Warning System: Identifies early-stage signs of diabetic retinopathy, allowing for timely medical intervention.
  3. Customizable and Extensible: Can be adapted to incorporate additional retinal conditions or advanced diagnostic features.

## How It Works
- Preprocessing of retinal images
- Deep learning model for detection
- Evaluation metrics for performance
- Visualization of results

## Tech Stack & Tools Used
- Programming Language: Python
- Deep Learning Framework: PyTorch
- Data Processing: OpenCV, PIL, Pandas, NumPy, Scikit-learn
- Visualization: Matplotlib, Seaborn
- Interface: Gradio
- Model Architecture: ResNet-152
- Dataset: APTOS 2019 Blindness Detection
- Hardware: GPU (CUDA-enabled recommended)

## Dataset & Model Architecture
The dataset used for this project can be obtained from the Kaggle Diabetic Retinopathy Detection competition. And the model used is trained using a ResNet-152 neural network architecture.

## Installation and setup
1. Clone the repository: git clone https://github.com/your-repo/blindness-detection.gitcd blindness-detection
2. Install required dependencies: pip install -r requirements.txt
3. Download the dataset and place it in the appropriate directory.
4. Run the training script: DRD_ResNet.ipynb
5. Launch the Gradio interface: interface.ipynb

## Usage
1. Run the interface.ipynb script to start the Gradio-based web application.
2. Upload a retinal image to get a prediction on diabetic retinopathy severity.

## Results
The model achieved a training accuracy of 96.01% and a validation accuracy of 93.2%, with a minimal loss of 0.136. On the test set, the accuracy was 92.32% with a loss of 0.149. 
