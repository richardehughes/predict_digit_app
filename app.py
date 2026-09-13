import joblib
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(page_title="Multi-Class Digit Classifier", layout="centered")

# Custom CSS to shrink button padding
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

st.markdown("### Multi-Class Digit Classifier")

# ==============================================================================
# 2. STATE MANAGEMENT & MODEL LOADING
# ==============================================================================
if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = 0

MODEL_FILE = "models/svc_mnist_digit_classifier.joblib"


@st.cache_resource
def load_classifier(model_path):
    """Loads pre-trained Multi-class SVC model from disk."""
    return joblib.load(model_path)


estimator = load_classifier(MODEL_FILE)


# ==============================================================================
# 3. HELPER FUNCTION: BOUNDING BOX CENTERING
# ==============================================================================
def center_digit_image(img_28x28):
    """Centers a drawn digit inside the 28x28 frame using bounding-box alignment.

    MNIST models perform best when strokes are centered rather than drawn in
    corners.
    """
    # Step 1: Find coordinates of non-background pixels (threshold brightness > 0.05)
    rows, cols = np.where(img_28x28 > 0.05)

    # Return original image if canvas is blank
    if len(rows) == 0 or len(cols) == 0:
        return img_28x28

    # Step 2: Find the bounding box boundaries of the drawing
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    # Step 3: Crop only the drawn region
    digit_crop = img_28x28[row_min : row_max + 1, col_min : col_max + 1]
    crop_h, crop_w = digit_crop.shape

    # Step 4: Calculate offsets to place the crop in the center of a blank 28x28 frame
    start_row = (28 - crop_h) // 2
    start_col = (28 - crop_w) // 2

    # Step 5: Create blank black grid and paste cropped digit into the middle
    centered_img = np.zeros((28, 28), dtype=np.float32)
    centered_img[
        start_row : start_row + crop_h, start_col : start_col + crop_w
    ] = digit_crop

    return centered_img


# ==============================================================================
# 4. USER INTERFACE (SIDEBAR & CANVAS)
# ==============================================================================
stroke_width = st.sidebar.slider("Brush Size", 10, 40, 25)
enable_centering = st.sidebar.checkbox("Enable Auto-Centering", value=True)

st.write("Draw a single digit (0-9) below:")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=f"digit_canvas_{st.session_state['canvas_key']}",
    update_streamlit=True,
    return_image_data=True,
)

col1, col2, _ = st.columns([0.2, 0.25, 0.55])

with col1:
    predict_clicked = st.button("Predict", type="primary")

with col2:
    if st.button("Clear Canvas"):
        st.session_state["canvas_key"] += 1
        st.rerun()

# ==============================================================================
# 5. PREPROCESSING & INFERENCE LOGIC
# ==============================================================================
if predict_clicked:
    if canvas_result.image_data is not None:
        # STEP A: Extract raw 280x280 RGBA array from canvas
        rgba_array = canvas_result.image_data.astype("uint8")

        # STEP B: Convert to Grayscale ('L') and downsample to 28x28
        img = Image.fromarray(rgba_array)
        img_gray = img.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
        raw_28x28 = np.array(img_gray) / 255.0

        # STEP C: Apply Centering Preprocessing (if toggled)
        if enable_centering:
            final_28x28 = center_digit_image(raw_28x28)
        else:
            final_28x28 = raw_28x28

        # STEP D: Flatten 28x28 matrix to (1, 784) for sklearn estimator input
        df_test_features = final_28x28.reshape(1, -1)

        # STEP E: Run Model Prediction & Decision Scores
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        predicted_class = estimator.predict(df_test_features)[0]
        decision_scores = estimator.decision_function(df_test_features)[0]

        # STEP F: Display Results & Image Preview
        st.success(f"### Predicted Digit: **{predicted_class}**")

        # Visual preview of what the model actually receives
        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            st.write("**Model Input (28x28 Grid):**")
            st.image(final_28x28, width=140, clamp=True)

        with preview_col2:
            st.write("**Decision Scores:**")
            best_digit = int(predicted_class)
            st.write(
                f"Top Digit ({best_digit}) Score: `{decision_scores[best_digit]:.3f}`"
            )

        with st.expander("See All Class Decision Scores"):
            for digit in labels:
                score = decision_scores[digit]
                st.write(f"**Digit {digit}:** Score = `{score:.3f}`")
    else:
        st.warning("Please draw a digit on the canvas before predicting.")