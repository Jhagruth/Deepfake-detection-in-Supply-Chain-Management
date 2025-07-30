import os
import numpy as np
import librosa
import tensorflow as tf
import warnings

# Suppress warnings from librosa
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

# --- 1. Configuration (MUST MATCH THE TRAINING SCRIPT) ---
SAMPLE_RATE = 16000
DURATION = 5  # seconds
N_MELS = 128
MODEL_FILENAME = 'deepfake_detector.keras' # The saved model file

# --- 2. Load the Trained Model ---
print(f"Loading pre-trained model from {MODEL_FILENAME}...")
if not os.path.exists(MODEL_FILENAME):
    print(f"Error: Model file not found at '{MODEL_FILENAME}'.")
    print("Please make sure you have run the training script first and the model file is in the same directory.")
    exit()

try:
    model = tf.keras.models.load_model(MODEL_FILENAME)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# --- 3. Feature Extraction Function (Copied from training script for consistency) ---
def extract_features(file_path):
    """Extracts a Mel spectrogram from a single audio file."""
    try:
        # Load audio file
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Pad or truncate to the fixed duration
        if len(audio) < DURATION * SAMPLE_RATE:
            audio = np.pad(audio, (0, DURATION * SAMPLE_RATE - len(audio)), 'constant')
        else:
            audio = audio[:DURATION * SAMPLE_RATE]
        
        # Compute Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
        
        # Convert to decibels
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        return log_mel_spectrogram
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- 4. Main Prediction Function ---
def verify_audio(file_path):
    """
    Loads an audio file, preprocesses it, and predicts if it's real or fake.
    """
    if not os.path.exists(file_path):
        print(f"Error: Audio file not found at '{file_path}'")
        return

    print(f"\nVerifying audio file: {os.path.basename(file_path)}")
    
    # Extract features from the new audio file
    features = extract_features(file_path)
    
    if features is None:
        print("Could not extract features from the audio file.")
        return
        
    # Reshape features to match the model's input shape: (1, n_mels, n_frames, 1)
    # The '1' at the beginning is the batch size.
    features = features.reshape(1, features.shape[0], features.shape[1], 1)
    
    # Get the model's prediction
    # The output is a probability value between 0 and 1 from the sigmoid function
    prediction_prob = model.predict(features)[0][0]
    
    # Interpret the result
    # We use a threshold of 0.5. If prob > 0.5, it's classified as 'spoof' (fake).
    # 'bonafide' was encoded as 0, 'spoof' as 1 during training.
    print(f"Model's Raw Prediction Score: {prediction_prob:.4f}")
    if prediction_prob > 0.5:
        print("Prediction: This audio is likely FAKE (spoofed).")
    else:
        print("Prediction: This audio is likely REAL (bonafide).")


# --- 5. Example Usage ---
if __name__ == '__main__':
    # IMPORTANT: Replace this with the path to the audio file you want to test.
    # It can be a .wav, .flac, or .mp3 file.
    path_to_your_audio = '/Users/jhagruth/Downloads/Yelahanka.m4a'
    
    verify_audio(path_to_your_audio)
    
    # You can test another file like this:
    # path_to_another_audio = 'another_audio.mp3'
    # verify_audio(path_to_another_audio)

