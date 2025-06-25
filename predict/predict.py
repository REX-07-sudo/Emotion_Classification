import streamlit as st
import librosa
import numpy as np
import joblib
import os
import tempfile

BASE_DIR = os.path.dirname(__file__)  
model_path = os.path.join(BASE_DIR, "emotion_model_7class.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler_7class.pkl")
le_path = os.path.join(BASE_DIR, "label_encoder_7class.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
le = joblib.load(le_path)

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

st.title("Speech Emotion Classifier")
st.markdown("Upload a `.wav` file to detect the emotion from speech.")

audio_file = st.file_uploader("Choose a WAV file", type=["wav"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    features = extract_features(tmp_path)
    os.remove(tmp_path)

    if features is not None:
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        emotion = le.inverse_transform([prediction])[0]

        st.success(f"Predicted Emotion: **{emotion.upper()}**")

        st.audio(audio_file, format='audio/wav')
