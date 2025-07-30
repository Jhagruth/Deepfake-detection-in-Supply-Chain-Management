import os
import kagglehub
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings from librosa about audioread
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

# --- 1. Download Dataset from Kaggle Hub ---
print("Connecting to Kaggle Hub to access the dataset...")
# This command points to the dataset cache. It will download on the first run.
# Ensure your Kaggle API key (kaggle.json) is in the correct location (~/.kaggle/).
try:
    path = kagglehub.dataset_download("awsaf49/asvpoof-2019-dataset")
    print(f"Dataset path: {path}")
except Exception as e:
    print(f"Error downloading dataset from Kaggle Hub: {e}")
    print("Please ensure your Kaggle API token (kaggle.json) is set up correctly.")
    exit()


# --- 2. Define Paths based on Kaggle Dataset Structure ---
# The structure within this specific Kaggle dataset is nested.
ASV_AUDIO_PATH = os.path.join(path, 'LA', 'LA', 'ASVspoof2019_LA_train', 'flac')
ASV_PROTOCOL_PATH = os.path.join(path, 'LA', 'LA', 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.train.trn.txt')

# --- 3. Configuration ---
SAMPLE_RATE = 16000 # ASVspoof uses 16kHz
DURATION = 5  # seconds to analyze per audio file
N_MELS = 128  # Number of Mel frequency bands
# Limit files to load for a quicker run. Set to None to load all files.
# Loading all files will take a very long time and require significant RAM.
MAX_FILES_TO_LOAD = 3000 

# --- 4. Feature Extraction and Data Loading ---
def extract_features(file_path):
    """Extracts a Mel spectrogram from an audio file."""
    try:
        # Load audio file
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Pad or truncate the audio to a fixed length
        if len(audio) < DURATION * SAMPLE_RATE:
            audio = np.pad(audio, (0, DURATION * SAMPLE_RATE - len(audio)), 'constant')
        else:
            audio = audio[:DURATION * SAMPLE_RATE]
        
        # Compute Mel spectrogram
        # CORRECTED THE TYPO HERE: sr=SAMPLE_RATE instead of sr=SAMPLE_rate
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
        
        # Convert to decibels (log scale)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        return log_mel_spectrogram
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_data_from_protocol(audio_path, protocol_path, max_files=None):
    """Loads audio data by parsing the ASVspoof protocol file."""
    print("Loading data from ASVspoof protocol file...")
    
    # Check if protocol file exists
    if not os.path.exists(protocol_path):
        print(f"ERROR: Protocol file not found at {protocol_path}")
        return None, None

    protocol = pd.read_csv(protocol_path, sep=' ', header=None)
    protocol.columns = ['speaker_id', 'filename', 'system_id', 'key', 'label']

    features = []
    labels = []
    
    file_count = 0
    for _, row in protocol.iterrows():
        if max_files and file_count >= max_files:
            print(f"Reached file limit of {max_files}.")
            break
        
        filename = row['filename']
        label = row['label']
        
        file_path = os.path.join(audio_path, f"{filename}.flac")
        
        if os.path.exists(file_path):
            data = extract_features(file_path)
            if data is not None:
                features.append(data)
                labels.append(label)
                file_count += 1
                if file_count % 500 == 0:
                    print(f"Processed {file_count} files...")
        else:
            # This warning is common if MAX_FILES_TO_LOAD is set, not an error.
            # print(f"Warning: File not found {file_path}")
            pass

    if not features:
        print("ERROR: No features were extracted. Check your paths and file integrity.")
        return None, None

    return np.array(features), np.array(labels)

# Load the data
features, labels = load_data_from_protocol(ASV_AUDIO_PATH, ASV_PROTOCOL_PATH, max_files=MAX_FILES_TO_LOAD)

if features is None:
    print("Could not load data. Exiting.")
    exit()

# Preprocess labels ('bonafide' -> 0, 'spoof' -> 1)
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Reshape features for the CNN (add a channel dimension)
features = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

print(f"\nData loaded successfully.")
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# --- 5. Build and Train the CNN Model ---
print("\nBuilding the CNN model...")
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2], 1)),
    
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Increased dropout for more regularization
    
    Dense(1, activation='sigmoid') # Binary classification output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

print("\nStarting model training...")
history = model.fit(
    X_train, 
    y_train, 
    epochs=15, 
    batch_size=32, 
    validation_data=(X_test, y_test)
)

# --- 6. Save the Trained Model ---
MODEL_FILENAME = 'deepfake_detector.keras'
print(f"\nTraining complete. Saving model to {MODEL_FILENAME}...")
model.save(MODEL_FILENAME)
print(f"Model successfully saved.")

# --- 7. Evaluate the Model ---
print("\n--- Final Evaluation on Test Data ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
