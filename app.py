
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image, ImageOps, ImageFilter

st.set_page_config(
    page_title="DigitVision | AI Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

# ------------------------------
# CUSTOM CSS
# ------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #101827, #202d42);
    color: white;
}

h1, h2, h3 {
    color: #ffffff;
}

.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(120deg, #243b55, #141e30);
    margin-bottom: 22px;
}

.result-card {
    background: #25344a;
    padding: 24px;
    border-radius: 16px;
    text-align: center;
}

.result-number {
    font-size: 75px;
    font-weight: bold;
    color: #62d5fa;
}

.stButton button {
    background-color: #1976d2;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------
# LOAD TRAINED MODEL
# ------------------------------

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "handwritten_digit_cnn.keras"
    )

model = load_model()


# ------------------------------
# IMAGE PREPROCESSING
# ------------------------------

def preprocess_uploaded_image(image):

    image = image.convert("L")

    arr = np.array(image)

    border = np.concatenate([
        arr[0, :],
        arr[-1, :],
        arr[:, 0],
        arr[:, -1]
    ])

    if np.median(border) > 127:
        image = ImageOps.invert(image)

    image = image.filter(
        ImageFilter.MedianFilter(size=3)
    )

    arr = np.array(image)
    arr[arr < 35] = 0

    image = Image.fromarray(arr)

    bbox = image.getbbox()

    if bbox is None:
        raise ValueError("No digit detected in the image.")

    digit = image.crop(bbox)

    digit.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "L",
        (28, 28),
        0
    )

    x = (28 - digit.width) // 2
    y = (28 - digit.height) // 2

    canvas.paste(digit, (x, y))

    # Center based on the digit's intensity
    pixels = np.array(canvas, dtype=np.float32)

    total = pixels.sum()

    if total > 0:

        yy, xx = np.indices(pixels.shape)

        cx = (xx * pixels).sum() / total
        cy = (yy * pixels).sum() / total

        shift_x = int(round(13.5 - cx))
        shift_y = int(round(13.5 - cy))

        canvas = Image.new("L", (28, 28), 0)

        canvas.paste(
            Image.fromarray(pixels.astype("uint8")),
            (shift_x, shift_y)
        )

    return canvas


# ------------------------------
# HEADER
# ------------------------------

st.markdown("""
<div class="hero">
    <h1>🔢 DigitVision</h1>
    <h3>AI-Powered Handwritten Digit Recognition</h3>
    <p>
        Upload a handwritten digit image and let our
        Convolutional Neural Network recognize it.
    </p>
</div>
""", unsafe_allow_html=True)


# ------------------------------
# MODEL INFORMATION
# ------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Digit Classes", "10")
col2.metric("Input Resolution", "28 × 28")
col3.metric("Model", "CNN")


st.divider()


# ------------------------------
# IMAGE UPLOAD
# ------------------------------

st.header("Upload Your Handwritten Digit")

uploaded_file = st.file_uploader(
    "Choose a PNG, JPG or JPEG image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    original_image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            original_image,
            caption="Uploaded image",
            use_container_width=True
        )

    try:

        processed_image = preprocess_uploaded_image(
            original_image
        )

        with col2:

            st.subheader("Processed Image")

            st.image(
                processed_image,
                caption="28 × 28 model input",
                width=220
            )

        if st.button("Recognize Digit", use_container_width=True):

            input_array = (
                np.array(
                    processed_image,
                    dtype="float32"
                ) / 255.0
            )

            input_array = input_array.reshape(
                1, 28, 28, 1
            )

            probabilities = model.predict(
                input_array,
                verbose=0
            )[0]

            predicted_digit = int(
                np.argmax(probabilities)
            )

            confidence = float(
                np.max(probabilities) * 100
            )

            st.divider()

            st.subheader("Recognition Result")

            st.markdown(
                f"""
                <div class="result-card">
                    <p>Predicted Digit</p>
                    <div class="result-number">
                        {predicted_digit}
                    </div>
                    <p>Confidence: {confidence:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("Prediction Scores")

            probability_df = pd.DataFrame({
                "Digit": list(range(10)),
                "Confidence (%)": probabilities * 100
            })

            probability_df = probability_df.sort_values(
                "Confidence (%)",
                ascending=False
            )

            st.bar_chart(
                probability_df.set_index("Digit")
            )

            st.caption(
                "Scores are the CNN's softmax outputs, "
                "not guaranteed probabilities of correctness."
            )

    except ValueError as error:

        st.error(str(error))


# ------------------------------
# FOOTER
# ------------------------------

st.divider()

st.markdown(
    "DigitVision | Handwritten Digit Recognition "
    "using Convolutional Neural Networks"
)
