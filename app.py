import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = (180, 180)   # change 224 to YOUR image size from Step 2

@st.cache_resource
def load_everything():
    model = tf.keras.models.load_model("fruit_veg_model.keras")
    class_names = json.load(open("class_names.json"))
    return model, class_names

model, class_names = load_everything()

st.title("🍎 Fruit & Vegetable Classifier")
st.write("Upload a photo and the model will guess what it is.")

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, width=300)

x = np.array(img.resize(IMG_SIZE), dtype="float32") / 255.0
x = np.expand_dims(x, 0)

probs = model.predict(x)[0]
probs = tf.nn.softmax(probs).numpy()
best = int(np.argmax(probs))

    st.success(f"Prediction: {class_names[best]} ({probs[best]:.1%})")
