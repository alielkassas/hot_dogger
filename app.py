# Use a pipeline as a high-level helper
from transformers import pipeline
import streamlit as st
from PIL import Image
#from ultralytics import YOLO

#model = YOLO("best.pt")

pipe = pipeline("image-classification", model="julien-c/hotdog-not-hotdog")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)

    predictions = pipe(image)
    for pred in predictions:
        st.write(pred)



