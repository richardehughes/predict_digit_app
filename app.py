import os
import joblib
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="OvA Digit Classifier", layout="centered")
st.title("One-vs-All Digit Classifier")

# Update this path for local OSC runs, or use relative path for Streamlit Cloud
BASE_DIR = "models"  # e.g., "/users/PAS1043/osu7903/work/.../models/"


@st.cache_resource
def load_ova_models(model_dir):
    models = {}
    for digit in range(10):
        file_path = os.path.join(
            model_dir, f"mnist_svc_digit_{digit}_vs_all.joblib"
        )
        models[digit] = joblib.load(file_path)
    return models


# Load all 10 models once
models_by_digit = load_ova_models(BASE_DIR)

# Sidebar
stroke_width = st.sidebar.slider("Brush Size", 10, 40, 25)

# Drawing Canvas
st.write("Draw a single digit (0-9) below:")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="ova_canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Downsample drawing to 28x28 normalized array
        rgba_array = canvas_result.image_data.astype("uint8")
        img = Image.fromarray(rgba_array)
        img_gray = img.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
        features = (np.array(img_gray) / 255.0).reshape(1, -1)

        # Collect binary votes and decision boundary distances
        binary_votes = {}
        scores = {}
        signals_detected = 0

        for digit_signal in range(10):
            model = models_by_digit[digit_signal]
            pred = model.predict(features)[0]
            score = model.decision_function(features)[0]

            binary_votes[digit_signal] = pred
            scores[digit_signal] = score

            if pred == "signal":
                signals_detected += 1

        # Determine winner by highest decision score
        best_digit = max(scores, key=scores.get)

        # Output Primary Result
        if signals_detected == 0:
            st.warning(
                f"No binary model triggered 'signal'. **Closest match: Digit {best_digit}**"
            )
        else:
            st.success(
                f"### Predicted Digit: **{best_digit}** ({signals_detected} model(s) claimed signal)"
            )

        # Model Vote Breakdown
        with st.expander("See individual model votes & scores"):
            for digit in range(10):
                vote = binary_votes[digit]
                dist = scores[digit]
                st.write(
                    f"**Digit {digit} Model:** Vote = `{vote}` | Confidence Distance = `{dist:.3f}`"
                )
    else:
        st.warning("Please draw a digit before predicting.")