import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime


# ============================================================
# 🖼️ AI IMAGE CLASSIFICATION SYSTEM
# 👩‍💻 Developed by: Tehmina Anwar
# ============================================================

st.set_page_config(
    page_title="AI Image Classification System",
    page_icon="🖼️",
    layout="wide"
)


# ============================================================
# 🎨 CUSTOM STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.3);
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .prediction-title {
        font-size: 20px;
        font-weight: 600;
    }

    .footer {
        text-align: center;
        padding: 20px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 🖼️ HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🖼️ AI Image Classification System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">👩‍💻 Developed by Tehmina Anwar</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload an image or use your camera to identify objects "
    "using a pre-trained CNN model."
)

st.divider()


# ============================================================
# 🤖 LOAD MOBILENETV2 MODEL
# ============================================================

@st.cache_resource(show_spinner="🤖 Loading AI model...")
def load_model():

    model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=True
    )

    return model


try:

    model = load_model()
    model_status = True

except Exception as e:

    model = None
    model_status = False

    st.error("❌ AI model could not be loaded.")

    st.info(
        "Please check your Streamlit deployment logs."
    )


# ============================================================
# 📜 SESSION HISTORY
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# 🔍 IMAGE CLASSIFICATION FUNCTION
# ============================================================

def classify_image(image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    img_array = np.array(
        image,
        dtype=np.float32
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = (
        tf.keras.applications.mobilenet_v2
        .preprocess_input(img_array)
    )

    predictions = model.predict(
        img_array,
        verbose=0
    )

    decoded = (
        tf.keras.applications.mobilenet_v2
        .decode_predictions(
            predictions,
            top=3
        )[0]
    )

    return decoded


# ============================================================
# 💾 SAVE PREDICTION HISTORY
# ============================================================

def save_prediction(
    source,
    prediction,
    confidence
):

    st.session_state.history.append(
        {
            "Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Source": source,
            "Prediction": prediction,
            "Confidence": f"{confidence:.2f}%"
        }
    )


# ============================================================
# ⚙️ SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Options")

st.sidebar.markdown(
    """
### ✨ Features

📤 Image Upload

📷 Camera Detection

🤖 AI Classification

📊 Prediction Confidence

🏆 Top 3 Predictions

📜 Prediction History

📈 Result Dashboard

---

### 🛠️ Technology

🐍 Python

🧠 TensorFlow

🔬 CNN

👁️ Computer Vision

🎈 Streamlit

---

### 🤖 Model

**MobileNetV2**

**Dataset:** ImageNet
"""
)


# ============================================================
# 📷 IMAGE SOURCE
# ============================================================

input_method = st.radio(
    "📷 Choose Image Source",
    [
        "📤 Upload Image",
        "📷 Camera"
    ],
    horizontal=True
)


# ============================================================
# 📤 UPLOAD IMAGE
# ============================================================

if input_method == "📤 Upload Image":

    st.subheader("📤 Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
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

        # ----------------------------------------------------
        # IMAGE PREVIEW
        # ----------------------------------------------------

        with col1:

            st.subheader("📷 Uploaded Image")

            st.image(
                image,
                use_container_width=True
            )

            st.caption(
                f"File: {uploaded_file.name}"
            )

        # ----------------------------------------------------
        # AI PREDICTION
        # ----------------------------------------------------

        with col2:

            st.subheader("🤖 AI Prediction")

            if model_status:

                classify_button = st.button(
                    "🔍 Classify Image",
                    use_container_width=True
                )

                if classify_button:

                    with st.spinner(
                        "🤖 AI is analyzing the image..."
                    ):

                        decoded = classify_image(
                            image
                        )

                    st.success(
                        "✅ Classification Complete!"
                    )

                    # ------------------------------------------------
                    # BEST PREDICTION
                    # ------------------------------------------------

                    label = decoded[0][1]

                    confidence = (
                        decoded[0][2] * 100
                    )

                    display_label = (
                        label
                        .replace("_", " ")
                        .title()
                    )

                    # ------------------------------------------------
                    # RESULT
                    # ------------------------------------------------

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:

                        st.info(
                            f"""
                            🎯 **Predicted Class**

                            ## {display_label}
                            """
                        )

                    with result_col2:

                        st.success(
                            f"""
                            📊 **Confidence**

                            ## {confidence:.2f}%
                            """
                        )

                    # ------------------------------------------------
                    # TOP 3 PREDICTIONS
                    # ------------------------------------------------

                    st.subheader(
                        "🏆 Top 3 Predictions"
                    )

                    for rank, (_, name, score) in enumerate(
                        decoded,
                        start=1
                    ):

                        formatted_name = (
                            name
                            .replace("_", " ")
                            .title()
                        )

                        percentage = (
                            score * 100
                        )

                        st.write(
                            f"**{rank}. {formatted_name}**"
                        )

                        st.progress(
                            float(score)
                        )

                        st.caption(
                            f"Confidence: "
                            f"{percentage:.2f}%"
                        )

                    # ------------------------------------------------
                    # SAVE HISTORY
                    # ------------------------------------------------

                    save_prediction(
                        uploaded_file.name,
                        display_label,
                        confidence
                    )

            else:

                st.warning(
                    "⚠️ AI model is currently unavailable."
                )


# ============================================================
# 📷 CAMERA DETECTION
# ============================================================

else:

    st.subheader(
        "📷 Camera Detection"
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

        if model_status:

            camera_button = st.button(
                "🤖 Classify Camera Image",
                use_container_width=True
            )

            if camera_button:

                with st.spinner(
                    "🤖 AI is analyzing the camera image..."
                ):

                    decoded = classify_image(
                        image
                    )

                st.success(
                    "✅ Camera Image Classified!"
                )

                # ------------------------------------------------
                # BEST PREDICTION
                # ------------------------------------------------

                label = decoded[0][1]

                confidence = (
                    decoded[0][2] * 100
                )

                display_label = (
                    label
                    .replace("_", " ")
                    .title()
                )

                # ------------------------------------------------
                # RESULT
                # ------------------------------------------------

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    st.info(
                        f"""
                        🎯 **Predicted Class**

                        ## {display_label}
                        """
                    )

                with result_col2:

                    st.success(
                        f"""
                        📊 **Confidence**

                        ## {confidence:.2f}%
                        """
                    )

                # ------------------------------------------------
                # TOP 3
                # ------------------------------------------------

                st.subheader(
                    "🏆 Top 3 Predictions"
                )

                for rank, (_, name, score) in enumerate(
                    decoded,
                    start=1
                ):

                    formatted_name = (
                        name
                        .replace("_", " ")
                        .title()
                    )

                    percentage = (
                        score * 100
                    )

                    st.write(
                        f"**{rank}. {formatted_name}**"
                    )

                    st.progress(
                        float(score)
                    )

                    st.caption(
                        f"Confidence: "
                        f"{percentage:.2f}%"
                    )

                # ------------------------------------------------
                # SAVE HISTORY
                # ------------------------------------------------

                save_prediction(
                    "Camera Image",
                    display_label,
                    confidence
                )

        else:

            st.warning(
                "⚠️ AI model is currently unavailable."
            )


# ============================================================
# 📜 PREDICTION HISTORY
# ============================================================

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
            f"🕒 **{item['Time']}**  |  "
            f"📷 {item['Source']}  |  "
            f"🎯 {item['Prediction']}  |  "
            f"📊 {item['Confidence']}"
        )

    # ------------------------------------------------------------
    # CLEAR HISTORY
    # ------------------------------------------------------------

    if st.button(
        "🗑️ Clear Prediction History"
    ):

        st.session_state.history = []

        st.rerun()


# ============================================================
# 📊 RESULT DASHBOARD
# ============================================================

st.divider()

st.header(
    "📊 Result Dashboard"
)

total_predictions = len(
    st.session_state.history
)

dash_col1, dash_col2, dash_col3 = st.columns(3)


# ------------------------------------------------------------
# TOTAL PREDICTIONS
# ------------------------------------------------------------

with dash_col1:

    st.info(
        f"""
        ### 🔢 Total Predictions

        ## {total_predictions}
        """
    )


# ------------------------------------------------------------
# LAST PREDICTION
# ------------------------------------------------------------

with dash_col2:

    if total_predictions > 0:

        last_prediction = (
            st.session_state.history[-1]["Prediction"]
        )

        st.info(
            f"""
            ### 🎯 Last Prediction

            **{last_prediction}**
            """
        )

    else:

        st.info(
            """
            ### 🎯 Last Prediction

            **None**
            """
        )


# ------------------------------------------------------------
# LAST CONFIDENCE
# ------------------------------------------------------------

with dash_col3:

    if total_predictions > 0:

        last_confidence = (
            st.session_state.history[-1]["Confidence"]
        )

        st.info(
            f"""
            ### 📊 Last Confidence

            **{last_confidence}**
            """
        )

    else:

        st.info(
            """
            ### 📊 Last Confidence

            **0%**
            """
        )


# ============================================================
# ℹ️ PROJECT INFORMATION
# ============================================================

st.divider()

st.header(
    "ℹ️ Project Information"
)

info_col1, info_col2 = st.columns(2)

with info_col1:

    st.markdown(
        """
**📌 Project:**  
AI Image Classification System

**👩‍💻 Developer:**  
Tehmina Anwar

**🤖 Model:**  
MobileNetV2

**📚 Dataset:**  
ImageNet
"""
    )

with info_col2:

    st.markdown(
        """
**🧠 Framework:**  
TensorFlow

**🎈 Interface:**  
Streamlit

**👁️ Computer Vision:**  
CNN

**🏆 Classification:**  
Top-3 Predictions
"""
    )


# ============================================================
# 🎯 PROJECT OBJECTIVES
# ============================================================

st.divider()

st.header(
    "🎯 Project Objectives"
)

objectives = [
    "🤖 Implement image classification using a pre-trained CNN",
    "🧠 Use MobileNetV2 for object recognition",
    "📷 Support image upload and camera input",
    "📊 Display prediction confidence",
    "🏆 Show Top-3 predictions",
    "📜 Maintain prediction history",
    "📈 Provide a simple result dashboard",
    "🎈 Build an interactive Streamlit application"
]

for objective in objectives:

    st.write(
        f"• {objective}"
    )


# ============================================================
# 💡 WHAT I LEARNED
# ============================================================

st.divider()

st.header(
    "💡 What I Learned"
)

st.write(
    """
Through this project, I gained practical experience in:

🐍 Python development  
🧠 TensorFlow  
🔬 Convolutional Neural Networks  
👁️ Computer Vision  
🤖 MobileNetV2  
📊 Image classification  
📷 Camera-based AI applications  
🎈 Streamlit development  
☁️ AI application deployment
"""
)


# ============================================================
# 🔮 FUTURE IMPROVEMENTS
# ============================================================

st.divider()

st.header(
    "🔮 Future Improvements"
)

future_features = [
    "🧠 Custom-trained image classification model",
    "📊 Custom dataset support",
    "🎯 More classification categories",
    "📷 Continuous camera detection",
    "📈 Advanced analytics",
    "💾 Export prediction history",
    "📱 Mobile-friendly interface",
    "☁️ Improved cloud deployment"
]

for feature in future_features:

    st.write(
        f"• {feature}"
    )


# ============================================================
# 👩‍💻 ABOUT ME
# ============================================================

st.divider()

st.header(
    "👩‍💻 About Me"
)

st.subheader(
    "Tehmina Anwar"
)

st.write(
    """
**BSAI Student | AI/ML Engineer | Python Developer**

I am a Bachelor of Science in Artificial Intelligence student
interested in building practical applications using Artificial
Intelligence, Machine Learning, Generative AI, Natural Language
Processing, and Computer Vision.
"""
)

st.write(
    "**🌟 Areas of Interest**"
)

st.write(
    """
🐍 Python  
🤖 Machine Learning  
🧠 Generative AI  
💬 Large Language Models  
🔎 Retrieval-Augmented Generation  
📝 Natural Language Processing  
👁️ Computer Vision  
🚀 AI Application Development
"""
)


# ============================================================
# 🔗 CONNECT WITH ME
# ============================================================

st.divider()

st.header(
    "🔗 Connect With Me"
)

st.markdown(
    """
💻 **GitHub:**  
https://github.com/Tehmina124

🔗 **LinkedIn:**  
https://www.linkedin.com/in/tehmina-anwar-77b8a8414/

🌐 **Portfolio:**  
https://tehmina-portfolio-five.vercel.app/
"""
)


# ============================================================
# 👩‍💻 FOOTER
# ============================================================

st.divider()

st.caption(
    "👩‍💻 Developed by Tehmina Anwar | "
    "AI/ML Engineer • Python Developer"
)

st.caption(
    "🖼️ AI Image Classification System | "
    "Python • TensorFlow • CNN • Computer Vision"
)

st.caption(
    "© 2026 Tehmina Anwar"
)
