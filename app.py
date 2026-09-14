import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

st.set_page_config(
    page_title="120-Breed Dog Classifier",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}
div[data-testid="stImage"] img {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

IMG_SIZE = (224, 224)


def clean_label(label):
    label = label.split("-", 1)[-1]
    label = label.replace("_", " ")
    return label.title()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dog_breed_model.keras")


@st.cache_data
def load_class_names():
    with open("class_names.txt", "r") as f:
        return [line.strip() for line in f]


model = load_model()
class_names = load_class_names()

st.title("120-Breed Dog Classifier")
st.caption("MobileNetV2 fine-tuned on Stanford Dogs Dataset")

uploaded_file = st.file_uploader(
    "Upload a dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=IMG_SIZE)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(predictions)[-3:][::-1]

    results = []
    for i in top_indices:
        label = clean_label(class_names[i])
        prob = predictions[i] * 100
        results.append((label, prob))

    col1, col2 = st.columns([0.7, 1.3], gap="large")

    with col1:
        st.image(img, width=260)

    with col2:
        top_label = results[0][0]
        top_prob = results[0][1]

        st.subheader(top_label)
        st.write(f"Confidence: {top_prob:.2f}%")

        for label, prob in results:
            st.write(f"{label} — {prob:.2f}%")
            st.progress(float(prob) / 100)

    st.caption("Visually similar breeds may occasionally be confused (e.g. Beagle vs Foxhound).")