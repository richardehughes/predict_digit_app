import joblib
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(page_title="Digit Classifier", layout="centered")

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

st.markdown("#### Digit Classifier")

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
from scipy.ndimage import center_of_mass


def center_digit_image(img_28x28):
    """Preprocesses drawn images to strictly match original MNIST specs:

    1. Fits digit inside a 20x20 box (preserving aspect ratio).
    2. Centers the digit using Center of Mass (intensity centroid).
    """
    # 1. Find bounding box of non-zero pixels
    rows, cols = np.where(img_28x28 > 0.05)
    if len(rows) == 0 or len(cols) == 0:
        return img_28x28

    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    # Crop digit
    crop = img_28x28[row_min : row_max + 1, col_min : col_max + 1]
    crop_h, crop_w = crop.shape

    # 2. Rescale crop to fit within a 20x20 box (MNIST standard)
    if crop_h > crop_w:
        new_h = 20
        new_w = max(1, int(round((crop_w / crop_h) * 20)))
    else:
        new_w = 20
        new_h = max(1, int(round((crop_h / crop_w) * 20)))

    crop_img = Image.fromarray((crop * 255).astype(np.uint8))
    resized_crop = crop_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_arr = np.array(resized_crop) / 255.0

    # 3. Place in center of a blank 28x28 canvas
    canvas_28x28 = np.zeros((28, 28), dtype=np.float32)
    start_r = (28 - new_h) // 2
    start_c = (28 - new_w) // 2
    canvas_28x28[
        start_r : start_r + new_h, start_c : start_c + new_w
    ] = resized_arr

    # 4. Shift image so its Center of Mass sits at (13.5, 13.5)
    cy, cx = center_of_mass(canvas_28x28)
    if not np.isnan(cy) and not np.isnan(cx):
        shift_y = int(round(13.5 - cy))
        shift_x = int(round(13.5 - cx))
        canvas_28x28 = np.roll(canvas_28x28, shift_y, axis=0)
        canvas_28x28 = np.roll(canvas_28x28, shift_x, axis=1)

    return canvas_28x28

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