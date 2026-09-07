"""Streamlit prototype for mango ripeness assessment.

The feature order in ``extract_features`` must match the order used when the
Random Forest model was trained: 96 HSV histogram values followed by the seven
log-scaled Hu moments.
"""

from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "hybrid_random_forest.pkl"
IMAGE_SIZE = (224, 224)
HSV_BINS = 32
CLAHE_CLIP_LIMIT = 1.0
S_MIN = 20
KERNEL_SIZE = 7
MORPH_ITERATIONS = 2
CLASS_LABELS = {
    0: "Unripe",
    1: "Partially Ripe",
    2: "Ripe",
}


def preprocess_image(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same BGR-to-HSV enhancement used during model training."""
    rgb_image = np.asarray(image.convert("RGB"))
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    resized_bgr = cv2.resize(bgr_image, IMAGE_SIZE)
    hsv_image = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
    _, _, value = cv2.split(hsv_image)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
    enhanced_hsv = cv2.merge([hsv_image[:, :, 0], hsv_image[:, :, 1], clahe.apply(value)])
    return resized_bgr, enhanced_hsv


def _mango_mask(hsv_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the training pipeline's cleaned mask and final contour."""
    mask = cv2.inRange(
        hsv_image,
        np.array([0, S_MIN, 20]),
        np.array([179, 255, 255]),
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No mango detected in the uploaded image.")
    largest_contour = max(contours, key=cv2.contourArea)
    mango_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mango_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    clean_mask = cv2.morphologyEx(
        mango_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=MORPH_ITERATIONS,
    )
    final_contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not final_contours:
        raise ValueError("No mango contour detected in the uploaded image.")
    final_contour = max(final_contours, key=cv2.contourArea)
    return clean_mask, final_contour


def extract_features(image: Image.Image) -> np.ndarray:
    """Extract 103 features: 96 HSV histogram values + 7 Hu moments."""
    _, hsv_image = preprocess_image(image)
    clean_mask, final_contour = _mango_mask(hsv_image)
    histogram_features = []
    histogram_ranges = [(0, 180), (0, 256), (0, 256)]
    for channel, histogram_range in enumerate(histogram_ranges):
        histogram = cv2.calcHist(
            [hsv_image], [channel], clean_mask, [HSV_BINS], list(histogram_range)
        )
        histogram_features.extend(cv2.normalize(histogram, histogram).flatten().tolist())

    moments = cv2.moments(final_contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_features = [-np.sign(value) * np.log10(abs(value)) if value else 0.0 for value in hu_moments]
    features = np.asarray(histogram_features + hu_features, dtype=np.float32)
    if features.shape != (103,):
        raise ValueError(f"Expected 103 features, got {features.shape[0]}")
    return features


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def display_label(model_class) -> str:
    if isinstance(model_class, (np.integer, int)):
        return CLASS_LABELS.get(int(model_class), str(model_class))
    normalized = str(model_class).strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "unripe": "Unripe",
        "partially ripe": "Partially Ripe",
        "partiallyripe": "Partially Ripe",
        "ripe": "Ripe",
    }
    return aliases.get(normalized, str(model_class))


def main() -> None:
    st.set_page_config(page_title="Mango Ripeness Assessment", page_icon="🥭", layout="centered")
    st.title("🥭 Mango Ripeness Assessment")
    st.caption("Hybrid image processing pipeline + Random Forest classifier")

    st.markdown(
        "Upload a mango image to estimate whether it is **Unripe**, "
        "**Partially Ripe**, or **Ripe**."
    )
    uploaded_file = st.file_uploader("Choose a mango image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    image = Image.open(uploaded_file)
    st.subheader("Uploaded Mango Preview")
    st.image(image, use_container_width=True)

    if not MODEL_PATH.exists():
        st.warning(f"Model file not found: `{MODEL_PATH.name}`")
        st.code("joblib.dump(rf, 'hybrid_random_forest.pkl')")
        st.caption("Place the exported model in the same folder as app.py, then reload the page.")
        return

    if st.button("Predict Ripeness", type="primary", use_container_width=True):
        try:
            model = load_model()
            features = extract_features(image)
            prediction = model.predict(features.reshape(1, -1))[0]
            label = display_label(prediction)

            st.subheader("Prediction Result")
            st.success(label)

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features.reshape(1, -1))[0]
                confidence = float(np.max(probabilities))
                st.metric("Prediction Confidence", f"{confidence:.2%}")
                st.write("Class Probabilities")
                for model_class, probability in zip(model.classes_, probabilities):
                    st.progress(float(probability), text=f"{display_label(model_class)} — {probability:.2%}")
            else:
                st.info("This model does not provide class probabilities.")
        except Exception as error:
            st.error("The image could not be processed with this model.")
            st.exception(error)

    with st.expander("Feature configuration"):
        st.write("103 features: 96 HSV histogram values (32 bins per channel) + 7 Hu moments.")
        st.write("Preprocessing: 224 × 224 resize, CLAHE on V, HSV thresholding, and morphological closing.")
        st.write("The separate visualization-only shadow-removal preprocessing is not used.")


if __name__ == "__main__":
    main()
