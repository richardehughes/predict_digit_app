import os
import joblib
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# 1. Page Config
st.set_page_config(page_title="OvA Digit Classifier", layout="centered")

# Custom CSS to shrink button padding and font size
st.markdown(
    """
    <style>
    div.stButton > button {
        padding: 2px 10px !important;
        font-size: 13px !important;
        min-height: 0px !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Smaller title (Markdown header instead of st.title)
st.markdown("### One-vs-All Digit Classifier")

# 2. State management for resetting the canvas
if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = 0

BASE_DIR = "models"


@st.cache_resource
def load_ova_models(model_dir):
    models = {}
    for digit in range(10):
        file_path = os.path.join(
            model_dir, f"mnist_svc_digit_{digit}_vs_all.joblib"
        )
        models[digit] = joblib.load(file_path)
    return models


models_by_digit = load_ova_models(BASE_DIR)

# 3. Sidebar Controls
stroke_width = st.sidebar.slider("Brush Size", 10, 40, 25)

st.write("Draw a single digit (0-9) below:")

# 4. Interactive Drawing Canvas (280x280)
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=f"ova_canvas_{st.session_state['canvas_key']}",
    update_streamlit=True,
    return_image_data=True,
)

# 5. Compact Action Buttons (narrow column allocations, default width)
col1, col2, _ = st.columns([0.2, 0.25, 0.55])

with col1:
    predict_clicked = st.button("Predict", type="primary")

with col2:
    if st.button("Clear Canvas"):
        st.session_state["canvas_key"] += 1
        st.rerun()

# 6. Extraction & Ensemble Prediction
if predict_clicked:
    if canvas_result.image_data is not None:
        rgba_array = canvas_result.image_data.astype("uint8")
        img = Image.fromarray(rgba_array)
        img_gray = img.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
        features = (np.array(img_gray) / 255.0).reshape(1, -1)

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

        best_digit = max(scores, key=scores.get)

        if signals_detected == 0:
            st.warning(
                f"No binary model triggered 'signal'. **Closest match: Digit {best_digit}**"
            )
        else:
            st.success(
                f"### Predicted Digit: **{best_digit}** ({signals_detected} model(s) claimed signal)"
            )

        with st.expander("See individual model votes & scores"):
            for digit in range(10):
                vote = binary_votes[digit]
                dist = scores[digit]
                st.write(
                    f"**Digit {digit} Model:** Vote = `{vote}` | Decision Distance = `{dist:.3f}`"
                )
    else:
        st.warning("Please draw a digit before predicting.")