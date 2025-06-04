from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import numpy as np
import librosa
import noisereduce as nr
from scipy.signal import butter, sosfiltfilt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import uuid

app = FastAPI()

# Configuration (from modelr.ipynb)
SAMPLE_RATE = 16000
N_MFCC = 13
N_MELS = 22
WINDOW = int(SAMPLE_RATE * 0.01)
HOP = int(SAMPLE_RATE * 0.005)
COMMANDS = ["baca", "berhenti", "foto", "halo", "info", "kembali", "ulang"]

# Load the trained model
model = load_model('model/mymodelr.h5')

# Initialize LabelEncoder
le = LabelEncoder()
le.fit(COMMANDS)

# Preprocessing functions (from modelr.ipynb)
def load_audio(file_path, sr=SAMPLE_RATE):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio / np.max(np.abs(audio))

def bandpass_filter(audio, lowcut=300, highcut=3400, sr=SAMPLE_RATE, order=5):
    sos = butter(order, [lowcut, highcut], btype='band', fs=sr, output='sos')
    return sosfiltfilt(sos, audio)

def reduce_noise(audio):
    return nr.reduce_noise(y=audio, sr=SAMPLE_RATE)

def extract_mfcc(signal):
    return librosa.feature.mfcc(
        y=signal,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_mels=N_MELS,
        n_fft=WINDOW,
        hop_length=HOP
    ).T

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.endswith(".wav"):
            raise HTTPException(status_code=400, detail="Only .wav files are supported")

        # Save the uploaded file temporarily
        temp_file_path = f"temp_{uuid.uuid4()}.wav"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Preprocess the audio
        audio = load_audio(temp_file_path)
        audio = bandpass_filter(audio)
        audio = reduce_noise(audio)
        mfcc = extract_mfcc(audio)

        # Pad the MFCC to match the model's input shape
        max_len = model.input_shape[1]
        mfcc_padded = pad_sequences([mfcc], maxlen=max_len, padding='post', dtype='float32')

        # Make prediction
        prediction = model.predict(mfcc_padded)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = le.inverse_transform([predicted_class])[0]
        confidence = float(prediction[0][predicted_class])

        # Clean up temporary file
        os.remove(temp_file_path)

        return {"command": predicted_label, "confidence": confidence}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))