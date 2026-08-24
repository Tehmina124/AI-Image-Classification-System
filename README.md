# 🖼️ AI Image Classification System

<p align="center">
  <img src="./AI IMAGE.png" width="100%" alt="AI Image Classification System Banner">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Computer%20Vision-CNN-green?style=for-the-badge" alt="Computer Vision">
  <img src="https://img.shields.io/badge/Streamlit-Framework-red?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/MobileNetV2-ImageNet-purple?style=for-the-badge" alt="MobileNetV2">
</p>

<p align="center">
  <b>🤖 Intelligent Image Classification using Deep Learning</b>
</p>

<p align="center">
  Developed by <b>Tehmina Anwar</b>
</p>

<p align="center">
  <a href="YOUR_LIVE_DEMO_LINK">
    🚀 <b>View Live Demo</b>
  </a>
</p>

---

## 📌 About the Project

**AI Image Classification System** is a Deep Learning-based Computer Vision application that identifies objects and image categories using a pre-trained **MobileNetV2 Convolutional Neural Network (CNN)** model.

The application allows users to either **upload an image** or **capture an image using their camera**. The AI model analyzes the image and provides the **Top-3 predictions along with confidence scores**.

The project is developed using **Python, TensorFlow, MobileNetV2, NumPy, Pillow, and Streamlit**.

---

## 🎯 Main Features

* 📤 Image Upload
* 📷 Camera Image Capture
* 🤖 AI Image Classification
* 🧠 Pre-trained MobileNetV2 Model
* 📚 ImageNet Dataset
* 🏆 Top-3 Predictions
* 📊 Prediction Confidence
* 📜 Prediction History
* 📈 Result Dashboard
* ⚡ Interactive Streamlit Interface
* ☁️ AI Application Deployment

---

## 🧠 How It Works

```text
📷 Upload / Capture Image
          ↓
🖼️ Convert Image to RGB
          ↓
📐 Resize Image to 224 × 224
          ↓
🔢 Convert Image into NumPy Array
          ↓
🧠 MobileNetV2 Preprocessing
          ↓
🤖 AI Model Prediction
          ↓
🏆 Decode Top-3 Predictions
          ↓
📊 Display Confidence Scores
          ↓
📜 Save Prediction History
```

---

## 🤖 Model

### MobileNetV2

The application uses the **MobileNetV2** deep learning architecture for image classification.

**Model:** MobileNetV2
**Framework:** TensorFlow / Keras
**Weights:** ImageNet
**Input Size:** 224 × 224 pixels
**Task:** Image Classification
**Output:** Top-3 Predictions

MobileNetV2 is a lightweight CNN architecture designed for efficient computer vision and image recognition tasks.

---

## 📚 Dataset

The application uses the pre-trained **ImageNet** weights provided with MobileNetV2.

The model can recognize a large number of common image categories without requiring the user to train a model from scratch.

---

## 📊 Prediction Results

After analyzing an image, the application displays:

| Result             | Description                      |
| ------------------ | -------------------------------- |
| 🎯 Predicted Class | Most likely image category       |
| 📊 Confidence      | Prediction confidence percentage |
| 🏆 Top 3           | Three most likely predictions    |

Example:

```text
🥇 Golden Retriever — 92.45%
🥈 Labrador Retriever — 4.21%
🥉 Dog — 1.86%
```

> Actual predictions depend on the image provided by the user.

---

## 📷 Camera Detection

The application also supports camera input.

Users can:

1. Select **📷 Camera**
2. Take a picture
3. Submit the image
4. Let the AI analyze it
5. View the Top-3 predictions
6. Check the confidence scores

```text
📷 Camera
    ↓
📸 Capture Image
    ↓
🤖 AI Analysis
    ↓
🏆 Top-3 Predictions
    ↓
📊 Confidence Scores
```

---

## 📜 Prediction History

The system maintains a session-based prediction history.

It records:

* 🕒 Prediction Time
* 📷 Image Source
* 🎯 Predicted Class
* 📊 Confidence Score

This allows users to review predictions made during their current session.

---

## 📈 Result Dashboard

The application includes a simple dashboard with:

* 🔢 Total Predictions
* 🎯 Last Prediction
* 📊 Last Confidence

This provides a quick overview of the classification activity.

---

## 🛠️ Technology Stack

| Technology              | Purpose                    |
| ----------------------- | -------------------------- |
| 🐍 **Python**           | Application Development    |
| 🧠 **TensorFlow**       | Deep Learning Framework    |
| 🤖 **MobileNetV2**      | Image Classification Model |
| 📚 **ImageNet**         | Pre-trained Dataset        |
| 🔢 **NumPy**            | Numerical Operations       |
| 🖼️ **Pillow**          | Image Processing           |
| 🎈 **Streamlit**        | Web Interface              |
| 👁️ **Computer Vision** | Image Recognition          |
| 🧠 **CNN**              | Deep Learning              |
| 🐙 **GitHub**           | Version Control            |
| ☁️ **Streamlit Cloud**  | Deployment                 |

---

## 📂 Project Structure

```text
AI-Image-Classification/
│
├── app.py
├── requirements.txt
├── AI IMAGE.png
└── README.md
```

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Tehmina124/AI-Image-Classification.git
```

### 2️⃣ Open the Project Folder

```bash
cd AI-Image-Classification
```

### 3️⃣ Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
python -m streamlit run app.py
```

### 5️⃣ Open in Browser

```text
http://localhost:8501
```

---

## 🌐 Live Demo

<p align="center">
  <a href="YOUR_LIVE_DEMO_LINK">
    🚀 <b>Open AI Image Classification System</b>
  </a>
</p>

The application can be deployed using **Streamlit Community Cloud**.

---

## 🎯 Project Objectives

* 🤖 Implement image classification using Deep Learning
* 🧠 Use a pre-trained CNN model
* 📷 Support image upload and camera input
* 🎯 Identify image categories
* 📊 Display prediction confidence
* 🏆 Show Top-3 predictions
* 📜 Maintain prediction history
* 📈 Create an interactive dashboard
* 🎈 Build an AI application using Streamlit
* ☁️ Deploy an AI application online

---

## 💡 What I Learned

Through this project, I gained practical experience in:

* 🐍 Python Development
* 🧠 TensorFlow
* 🔬 Convolutional Neural Networks
* 👁️ Computer Vision
* 🤖 MobileNetV2
* 📚 ImageNet
* 🖼️ Image Preprocessing
* 📊 Image Classification
* 📷 Camera-Based AI Applications
* 🎈 Streamlit Development
* 🐙 GitHub & Git
* ☁️ AI Application Deployment

---

## 🔮 Future Improvements

* 🧠 Custom-trained Image Classification Model
* 📊 Custom Dataset Support
* 🎯 More Classification Categories
* 📷 Continuous Real-Time Camera Detection
* 📈 Advanced Analytics Dashboard
* 💾 Export Prediction History
* 📱 Mobile-Friendly Interface
* 🧠 Transfer Learning with Custom Datasets
* 🎯 Object Detection Support
* ☁️ Improved Cloud Deployment

---

## 📚 Project Highlights

| 🤖 Feature              | 🎯 Description                   |
| ----------------------- | -------------------------------- |
| 📤 Image Upload         | Upload images for classification |
| 📷 Camera Input         | Capture images using camera      |
| 🧠 MobileNetV2          | Pre-trained CNN model            |
| 📚 ImageNet             | Model training dataset           |
| 🎯 Image Classification | Identify image categories        |
| 🏆 Top-3 Predictions    | Display three likely classes     |
| 📊 Confidence Score     | Show prediction confidence       |
| 📜 Prediction History   | Store session predictions        |
| 📈 Result Dashboard     | Display prediction statistics    |
| ⚡ Streamlit Interface   | Interactive web application      |
| ☁️ Cloud Deployment     | Online AI application            |

---

## 🔬 Classification Pipeline

```text
Original Image
      ↓
RGB Conversion
      ↓
Resize → 224 × 224
      ↓
NumPy Array
      ↓
MobileNetV2 Preprocessing
      ↓
CNN Model
      ↓
Prediction Probabilities
      ↓
Decode ImageNet Classes
      ↓
Top-3 Results
```

---

## 🎓 Learning Outcomes

This project helped me understand how to:

* Build an AI-powered Computer Vision application
* Use a pre-trained CNN model
* Apply image preprocessing
* Perform image classification
* Work with TensorFlow and Keras
* Display prediction confidence
* Build an interactive Streamlit interface
* Handle image uploads and camera input
* Maintain prediction history
* Deploy an AI application

---

## 👩‍💻 About Me

### **Tehmina Anwar**

**BSAI Student | AI/ML Engineer | Python Developer**

I am a **Bachelor of Science in Artificial Intelligence student** interested in building practical **AI and Machine Learning applications**.

### 🌟 Areas of Interest

* 🐍 Python
* 🤖 Machine Learning
* 🧠 Generative AI
* 💬 Large Language Models
* 🔎 Retrieval-Augmented Generation
* 📝 Natural Language Processing
* 👁️ Computer Vision
* 🚀 AI Application Development

---

## 🔗 Connect With Me

### 💻 GitHub

<a href="https://github.com/Tehmina124">
  <b>GitHub Profile</b>
</a>

### 🔗 LinkedIn

<a href="https://www.linkedin.com/in/tehmina-anwar-77b8a8414/">
  <b>LinkedIn Profile</b>
</a>

### 🌐 Portfolio

<a href="https://tehmina-portfolio-five.vercel.app/">
  <b>Portfolio Website</b>
</a>

---

## ⭐ Support

If you found this project useful or interesting, consider giving the repository a **⭐ Star on GitHub**.

Your support is greatly appreciated! ❤️

---

<p align="center">
  <b>🤖 Built with ❤️ using Python, TensorFlow, Deep Learning & Streamlit</b>
</p>

<p align="center">
  © 2026 <b>Tehmina Anwar</b> | AI Image Classification System
</p>
