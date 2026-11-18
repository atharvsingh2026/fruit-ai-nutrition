from pathlib import Path
import io
import base64
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps


BASE = Path(__file__).resolve().parent
MODEL = BASE / 'model' / 'keras_model.h5'
LABELS = BASE / 'model' / 'labels.txt'
DATA = BASE / 'nutrition.csv'

st.set_page_config(page_title='AI Fruit Nutrition Scanner', page_icon='🍎', layout='wide')
st.title('🍎 AI Fruit Nutrition Scanner')
st.caption('Class 12 AI Capstone | Educational project — not medical advice' | 'Developed by Atharv Singh (Class 12-B2)')

@st.cache_resource
def load_classifier():
    from tensorflow.keras.models import load_model
    return load_model(str(MODEL), compile=False)

@st.cache_data
def load_nutrition():
    df = pd.read_csv(DATA).fillna('')
    return {str(row['fruit']).strip().lower(): row for _, row in df.iterrows()}

def read_labels():
    lines = LABELS.read_text(encoding='utf-8').splitlines()
    # Teachable Machine exports labels as '0 Apple', '1 Banana', etc.
    return [line.strip().split(' ', 1)[-1].strip() for line in lines if line.strip()]

def predict(image):
    model = load_classifier()
    labels = read_labels()
    image = ImageOps.fit(image.convert('RGB'), (224, 224), method=Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    scores = np.asarray(model.predict(np.expand_dims(arr, axis=0), verbose=0)[0])
    if len(scores) != len(labels):
        raise ValueError('Model outputs and labels.txt have different class counts.')
    index = int(np.argmax(scores))
    return labels[index], float(scores[index]), sorted(zip(labels, scores.tolist()), key=lambda x: x[1], reverse=True)

uploaded = st.file_uploader('Upload a clear fruit photo', type=['jpg', 'jpeg', 'png'])
threshold = st.sidebar.slider('Minimum confidence (%)', 50, 95, 70)
if uploaded:
    try:
        if uploaded.size > 10 * 1024 * 1024:
            st.error("Please upload an image smaller than 10 MB.")
            st.stop()

        # Read uploaded image
        image_bytes = uploaded.getvalue()

        # Create picture variable for AI prediction
        picture = Image.open(io.BytesIO(image_bytes))
        picture.load()
        picture = picture.convert("RGB")

        # Convert image to Base64
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=90)

        encoded_image = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        # Display uploaded image
        st.markdown(
            f"""
            <div style="margin:15px 0;">
                <img
                    src="data:image/jpeg;base64,{encoded_image}"
                    style="
                        width:320px;
                        max-width:100%;
                        height:auto;
                        border-radius:12px;
                    "
                />
                <p>Uploaded fruit image</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Keep your existing Analyze Fruit code below
        if st.button("🔍 Analyze Fruit", type="primary"):

                st.info("Analyzing fruit image...")

                if not MODEL.exists() or not LABELS.exists():
                    st.error(
                        "AI model files not found. "
                        "Please check the model folder."
                    )

                else:
                    with st.spinner("AI is analyzing the image..."):

                        name, confidence, predictions = predict(picture)

                    st.subheader("AI Prediction")

                    st.success(f"Detected Fruit: {name}")

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.1f}%"
                    )

                    if confidence * 100 < threshold:
                        st.warning(
                            "Low confidence. Please upload "
                            "a clearer fruit image."
                        )

                    else:
                        nutrition = load_nutrition().get(
                            name.lower()
                        )

                        if nutrition is None:
                            st.warning(
                                f"No nutrition data found for {name}"
                            )

                        else:
                            st.subheader(
                                "Nutrition per 100 g"
                            )

                            cols = st.columns(4)

                            fields = [
                                ("calories_kcal", "kcal"),
                                ("carbs_g", "g"),
                                ("protein_g", "g"),
                                ("fiber_g", "g")
                            ]

                            for col, (field, unit) in zip(
                                cols, fields
                            ):
                                col.metric(
                                    field.replace("_", " ").title(),
                                    f"{nutrition[field]} {unit}"
                                )

                            st.write(
                                f"**Vitamin C:** "
                                f"{nutrition['vitamin_c_mg']} mg"
                            )

                            st.write(
                                f"**Potassium:** "
                                f"{nutrition['potassium_mg']} mg"
                            )

                            st.info(
                                f"**Nutritional Benefits:** "
                                f"{nutrition['benefits']}"
                            )

                            st.write(
                                f"**Dietary Context:** "
                                f"{nutrition['dietary_context']}"
                            )

                    with st.expander("All Model Predictions"):
                        for label, score in predictions:
                            st.write(
                                f"{label}: {score * 100:.1f}%"
                            )
                            st.progress(
                                max(0.0, min(1.0, score))
                            )

    except Exception as exc:
        st.error(f"Could not analyze this image: {exc}")
else:
    st.info('Choose a JPG or PNG image to begin.')
st.divider()
st.caption('Values are approximate reference figures for raw edible fruit per 100 g; variety and ripeness can change them. For a school project only.')
