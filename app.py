import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime

# ============================================
# 🖼️ AI Image Classification System
# 👩‍💻 Author: Tehmina Anwar
# ============================================

st.set_page_config(
    page_title="AI Image Classification System",
    page_icon="🖼️",
    layout="wide"
)

# ============================================
# Load Pre-trained CNN Model
# ============================================

@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(
        weights="imagenet"
    )

model = load_model()

# ============================================
# Session History
# ============================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================
# Image Classification Function
# ============================================

def classify_image(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array
    )

    predictions = model.predict(
        img_array,
        verbose=0
    )

    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
        predictions,
        top=3
    )[0]

    return decoded


# ============================================
# Save Prediction History
# ============================================

def save_prediction(
    source,
    prediction,
    confidence
):

    st.session_state.history.append({

        "Time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "Source": source,

        "Prediction": prediction,

        "Confidence": f"{confidence:.2f}%"
    })


# ============================================
# HEADER
# ============================================

st.title(
    "🖼️ AI Image Classification System"
)

st.subheader(
    "👩‍💻 Developed by Tehmina Anwar"
)

st.write(
    "Upload an image or use your camera to identify "
    "objects using a pre-trained CNN model."
)

st.divider()


# ============================================
# SIDEBAR
# ============================================

st.sidebar.title(
    "⚙️ Options"
)

st.sidebar.info(
    """
### Features

📤 Image Upload

📷 Camera Detection

🤖 AI Classification

📊 Prediction Confidence

🏆 Top 3 Predictions

📜 History Tracking

📈 Result Dashboard

### Technology

🐍 Python

🧠 TensorFlow

🔬 CNN

👁️ Computer Vision
"""
)


# ============================================
# IMAGE SOURCE
# ============================================

input_method = st.radio(
    "Choose Image Source",
    [
        "📤 Upload Image",
        "📷 Camera"
    ],
    horizontal=True
)


# ============================================
# IMAGE UPLOAD
# ============================================

if input_method == "📤 Upload Image":

    uploaded_file = st.file_uploader(
        "📤 Upload an Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        col1, col2 = st.columns(2)

        # ----------------------------
        # Image
        # ----------------------------

        with col1:

            st.subheader(
                "📷 Uploaded Image"
            )

            st.image(
                image,
                use_container_width=True
            )

        # ----------------------------
        # Prediction
        # ----------------------------

        with col2:

            st.subheader(
                "🤖 AI Prediction"
            )

            if st.button(
                "🔍 Classify Image",
                use_container_width=True
            ):

                with st.spinner(
                    "🤖 AI is analyzing the image..."
                ):

                    decoded = classify_image(
                        image
                    )

                st.success(
                    "✅ Classification Complete!"
                )

                # Best prediction

                label = decoded[0][1]

                confidence = (
                    decoded[0][2] * 100
                )

                st.metric(
                    "🎯 Predicted Class",
                    label.replace(
                        "_",
                        " "
                    ).title()
                )

                st.metric(
                    "📊 Confidence",
                    f"{confidence:.2f}%"
                )

                # ----------------------------
                # Top 3 Predictions
                # ----------------------------

                st.subheader(
                    "🏆 Top 3 Predictions"
                )

                for rank, (_, name, score) in enumerate(
                    decoded,
                    start=1
                ):

                    st.write(
                        f"**{rank}. "
                        f"{name.replace('_', ' ').title()}**"
                    )

                    st.progress(
                        float(score)
                    )

                    st.write(
                        f"Confidence: "
                        f"{score * 100:.2f}%"
                    )

                # ----------------------------
                # Save History
                # ----------------------------

                save_prediction(
                    uploaded_file.name,
                    label.replace(
                        "_",
                        " "
                    ).title(),
                    confidence
                )


# ============================================
# CAMERA DETECTION
# ============================================

else:

    st.subheader(
        "📷 Real-Time Camera Detection"
    )

    st.write(
        "Take a picture using your camera "
        "and let the AI classify it."
    )

    camera_image = st.camera_input(
        "📸 Take a Picture"
    )

    if camera_image is not None:

        image = Image.open(
            camera_image
        ).convert("RGB")

        st.image(
            image,
            caption="📷 Camera Image",
            use_container_width=True
        )

        if st.button(
            "🤖 Classify Camera Image",
            use_container_width=True
        ):

            with st.spinner(
                "🤖 AI is analyzing the camera image..."
            ):

                decoded = classify_image(
                    image
                )

            st.success(
                "✅ Camera Image Classified!"
            )

            label = decoded[0][1]

            confidence = (
                decoded[0][2] * 100
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "🎯 Predicted Class",
                    label.replace(
                        "_",
                        " "
                    ).title()
                )

            with col2:

                st.metric(
                    "📊 Confidence",
                    f"{confidence:.2f}%"
                )

            # ----------------------------
            # Top 3 Camera Predictions
            # ----------------------------

            st.subheader(
                "🏆 Top 3 Predictions"
            )

            for rank, (_, name, score) in enumerate(
                decoded,
                start=1
            ):

                st.write(
                    f"**{rank}. "
                    f"{name.replace('_', ' ').title()}**"
                )

                st.progress(
                    float(score)
                )

                st.write(
                    f"Confidence: "
                    f"{score * 100:.2f}%"
                )

            # ----------------------------
            # Save Camera History
            # ----------------------------

            save_prediction(
                "Camera Image",
                label.replace(
                    "_",
                    " "
                ).title(),
                confidence
            )


# ============================================
# PREDICTION HISTORY
# ============================================

st.divider()

st.header(
    "📜 Prediction History"
)

if len(st.session_state.history) == 0:

    st.info(
        "No predictions yet. "
        "Upload an image or use the camera."
    )

else:

    for item in reversed(
        st.session_state.history
    ):

        st.write(
            f"🕒 **{item['Time']}** | "
            f"📷 {item['Source']} | "
            f"🎯 {item['Prediction']} | "
            f"📊 {item['Confidence']}"
        )


# ============================================
# RESULT DASHBOARD
# ============================================

st.divider()

st.header(
    "📊 Result Dashboard"
)

total_predictions = len(
    st.session_state.history
)

col1, col2, col3 = st.columns(3)

# Total

with col1:

    st.metric(
        "🔢 Total Predictions",
        total_predictions
    )

# Last Prediction

with col2:

    if total_predictions > 0:

        st.metric(
            "🎯 Last Prediction",
            st.session_state.history[-1][
                "Prediction"
            ]
        )

    else:

        st.metric(
            "🎯 Last Prediction",
            "None"
        )

# Last Confidence

with col3:

    if total_predictions > 0:

        st.metric(
            "📊 Last Confidence",
            st.session_state.history[-1][
                "Confidence"
            ]
        )

    else:

        st.metric(
            "📊 Last Confidence",
            "0%"
        )


# ============================================
# PROJECT INFORMATION
# ============================================

st.divider()

st.header(
    "ℹ️ Project Information"
)

info_col1, info_col2 = st.columns(2)

with info_col1:

    st.write(
        """
**Project:** AI Image Classification System

**Developer:** Tehmina Anwar

**Model:** MobileNetV2

**Dataset:** ImageNet
"""
    )

with info_col2:

    st.write(
        """
**Framework:** TensorFlow

**Interface:** Streamlit

**Computer Vision:** CNN

**Classification:** Top-3 Predictions
"""
    )


# ============================================
# FOOTER
# ============================================

st.divider()

st.caption(
    "👩‍💻 Developed by Tehmina Anwar | "
    "AI/ML Engineer • Python Developer"
)

st.caption(
    "🖼️ AI Image Classification System | "
    "Python • TensorFlow • CNN • Computer Vision"
)
