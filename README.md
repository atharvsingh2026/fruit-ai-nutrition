# AI-Based Fruit Recognition and Nutrition Information System
Class 12 AI Capstone Project — Atharv, Aniket, Ansh, Nikunj

## 1. Train the AI model (required)
1. Visit https://teachablemachine.withgoogle.com/ and choose **Get Started → Image Project → Standard image model**.
2. Create 10 classes named exactly: Apple, Banana, Orange, Mango, Grapes, Guava, Papaya, Pineapple, Strawberry, Watermelon.
3. Upload approximately 30–50 or more varied images per class, using photos you have permission to use. Include different angles, backgrounds, and lighting.
4. Click **Train Model**. Test with photos NOT used for training.
5. Choose **Export Model → Tensorflow → Keras → Download my model** (not TensorFlow.js).
6. Extract the ZIP. Copy `keras_model.h5` and `labels.txt` into this project's `model` folder. Make sure the label names match the nutrition CSV.

## 2. Run on Windows 11 / VS Code
Install 64-bit **Python 3.11** first. In VS Code terminal (PowerShell):

```powershell
cd path\to\fruit_ai_project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run without activation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The browser normally opens at http://localhost:8501.

## 3. Test cases for school report
- Upload a clear apple photo → expect Apple, confidence varies by trained model.
- Upload a banana photo from outside training set → expect Banana.
- Upload a blurry or unfamiliar image → if top confidence is below threshold, ask for clearer image.
- Change a nutrition CSV value → rerun/reload and verify it appears.

## 4. Explain in viva
- **Training data**: labeled examples of each fruit.
- **Classification**: model chooses the fruit class with highest predicted score.
- **Confidence**: model score, not a guarantee of correctness.
- **Preprocessing**: crop to 224×224 RGB and scale pixel values to -1 to +1.
- **Nutrition lookup**: pandas reads local CSV; no online database is needed.
- **Limitations**: only trained fruits, background/lighting effects, approximate nutrition data; no medical advice.

## Notes
- The ZIP does **not** include a trained model; it must be created in Teachable Machine.
- Approximate nutrition figures are illustrative reference values for raw edible portions per 100 g. Verify the values against USDA FoodData Central (https://fdc.nal.usda.gov/) before submitting as verified data; cultivar and ripeness vary.
- TensorFlow/Keras compatibility can vary between exported Teachable Machine models and installed versions. TensorFlow 2.15.1 + Python 3.11 is a practical starting point, not a guarantee for every export.
