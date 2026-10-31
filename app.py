import streamlit as st
import tensorflow as tf
import numpy as np
import json

from PIL import Image

# Load trained model
model = tf.keras.models.load_model(
    "best_crop_model.keras"
)

# Load class names
with open("crop_class_names.json", "r") as f:
    class_names = json.load(f)

# Page configuration
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌱"
)

# Application heading
st.title("🌱 Smart Crop Disease Detection System")

st.write(
    "Upload a plant leaf image to identify "
    "its crop disease category."
)

# Image upload
uploaded_file = st.file_uploader(
    "Upload Crop Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read uploaded image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Crop Leaf",
        use_container_width=True
    )

    # Resize image
    resized_image = image.resize((180, 180))

    # Convert into array
    img_array = np.array(resized_image)

    img_array = np.expand_dims(
        img_array, axis=0
    )

    # Make prediction
    prediction = model.predict(img_array)

    # Get predicted class
    index = np.argmax(prediction[0])

    disease = class_names[index]

    confidence = prediction[0][index] * 100

    # Display results
    st.subheader("Prediction Result")

    st.success(
        f"Predicted Category: {disease}"
    )

    st.write(
        f"Confidence Score: {confidence:.2f}%"
    )

    # Display class scores
    st.subheader("Class Prediction Scores")

    scores = {
        class_names[i]: float(prediction[0][i])
        for i in range(len(class_names))
    }

    st.bar_chart(scores)

    st.warning(
        "This is an educational image-classification "
        "tool. Confirm suspected disease with a "
        "qualified agricultural expert before treatment."
    )
