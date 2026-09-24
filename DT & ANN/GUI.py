import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras.models import load_model
from joblib import dump, load

# Placeholder for models (replace with actual functions)
def predict_decision_tree(image):
    clf = load('handwritten_DT_.joblib')
    image_array = np.reshape(image,(1,-1))
    image_array = image_array / 255.0 
    prediction = clf.predict(image_array)
    return prediction

def predict_neural_network(image):
    model = load_model('handwritten_ANN.h5')
    image_array = np.reshape(image, (1, 784))
    image_array = image_array / 255.0
    predictions = model.predict(image_array)
    return predictions

def main():
    # Title and description
    st.title("Digit Recognizer")
    st.write("Draw a digit on the canvas and choose a model for prediction.")

    # Drawing area
    canvas = st_canvas(
        stroke_width=st.slider("Stroke width: ", 10, 15, 10),
        stroke_color="#fff",
        background_color="#000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Model selection
    model_choice = st.selectbox("Choose a Model", ("Decision Tree", "Neural Network"))

    result = st.button(f"Predict using {model_choice}")
    
    # Predict button
    if result:
        # Convert drawing to image and process for model input
        image = Image.fromarray((canvas.image_data[:, :, 0]).astype(np.uint8))
        image = image.resize((28, 28))
        image_array = np.array(image)

        st.image(image_array, caption="User Drawn Digit", use_column_width=True)

        # Get prediction based on model choice
        if model_choice == "Decision Tree":
            prediction = predict_decision_tree(image_array)
            prediction=prediction[0]
        else:
            all_predictions = predict_neural_network(image_array)
            prediction = np.argmax(all_predictions)
        
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.01)
        
        st.markdown(f"<h3 style = 'text-align: center;'>Prediction : {prediction}<h3>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
