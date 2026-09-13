import joblib
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(page_title="Multi-Class Digit Classifier", layout="centered")

# Custom CSS to shrink button padding and make UI compact for classroom use
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

# Compact title
st.markdown("### Multi-Class Digit Classifier")

# ==============================================================================
# 2. STATE MANAGEMENT & MODEL LOADING
# ==============================================================================
# Canvas Key Trick: Streamlit rebuilds widgets when their 'key' changes.
# Incrementing 'canvas_key' forces Streamlit to clear and reset the drawing area.
if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = 0

MODEL_FILE = "models/svc_mnist_digit_classifier.joblib"


# @st.cache_resource prevents Streamlit from reloading the model file on every user interaction
@st.cache_resource
def load_classifier(model_path):
    """Loads the pre-trained Multi-class SVC model from disk."""
    return joblib.load(model_path)


# Load the single estimator model
estimator = load_classifier(MODEL_FILE)

# ==============================================================================
# 3. USER INTERFACE (SIDEBAR & CANVAS)
# ==============================================================================
# Sidebar slider allowing students to experiment with brush stroke thickness
stroke_width = st.sidebar.slider("Brush Size", 10, 40, 25)

st.write("Draw a single digit (0-9) below:")

# Interactive 280x280 Canvas widget
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",  # Transparent fill
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",  # White brush (MNIST standard)
    background_color="#000000",  # Black canvas (MNIST standard)
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=f"digit_canvas_{st.session_state['canvas_key']}",
    update_streamlit=True,
    return_image_data=True,  # Sends RGBA pixel array back to Python
)

# Compact Action Buttons using fractional columns
col1, col2, _ = st.columns([0.2, 0.25, 0.55])

with col1:
    predict_clicked = st.button("Predict", type="primary")

with col2:
    if st.button("Clear Canvas"):
        # Incrementing the key forces Streamlit to re-render a blank canvas
        st.session_state["canvas_key"] += 1
        st.rerun()

# ==============================================================================
# 4. PREPROCESSING & INFERENCE LOGIC
# ==============================================================================
if predict_clicked:
    if canvas_result.image_data is not None:
        # STEP A: Extract raw 280x280 RGBA array from canvas
        rgba_array = canvas_result.image_data.astype("uint8")

        # STEP B: Convert to Grayscale ('L') and downsample to 28x28 pixel grid
        # Resampling.LANCZOS creates smooth anti-aliasing similar to original MNIST dataset
        img = Image.fromarray(rgba_array)
        img_gray = img.convert("L").resize((28, 28), Image.Resampling.LANCZOS)

        # STEP C: Feature Normalization & Reshaping
        # 1. Convert pixels from integers [0, 255] to floats [0.0, 1.0]
        # 2. Flatten 28x28 matrix into 1D array of 784 features, shaped (1, 784) for sklearn
        df_test_features = (np.array(img_gray) / 255.0).reshape(1, -1)

        # STEP D: Run Model Prediction & Decision Scores
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # 1. Predict class label directly (returns integer 0-9)
        predicted_class = estimator.predict(df_test_features)[0]

        # 2. Get decision function scores across all 10 classes
        # For multi-class SVC, decision_function returns distance to hyperplanes for each class
        decision_scores = estimator.decision_function(df_test_features)[0]

        # STEP E: Display Results
        st.success(f"### Predicted Digit: **{predicted_class}**")

        # Display raw decision function scores in an expandable section
        with st.expander("See Class Decision Scores"):
            st.write(
                "Higher (more positive) scores indicate greater model confidence for that digit class:"
            )

            # Iterate over the 10 digit classes and show calculated confidence score
            for digit in labels:
                score = decision_scores[digit]
                st.write(f"**Digit {digit}:** Decision Score = `{score:.3f}`")
    else:
        st.warning("Please draw a digit on the canvas before predicting.")