import streamlit as st
import os
import subprocess
import tempfile
import base64
import mimetypes
from io import BytesIO
import numpy as np
import cv2
import concurrent.futures
import math
import requests
from textwrap import dedent
import warnings

# Suppress warnings from librosa and tensorflow (if any)
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow warnings

# --- Global Configuration (adjust as needed) ---
# For face_r.py
ROBOFLOW_API_URL = "http://localhost:9001" # <<<--- Ensure this points to your local Roboflow Inference Server
ROBOFLOW_API_KEY = "zWV15F4167MkKAlS4nVh" # Your Roboflow API Key
ROBOFLOW_MODEL_IDS = [
    "deepfake-yryoa/1",
    "deepfake-detection-v5wyo/1",
    "deepfake-apkbz/1"
]

# For receipt.py
GEMINI_API_KEY = "AIzaSyBh8f2N8VvKPJcsfH7k417xFnxlRyR8IpI"  # Your Gemini API Key
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# For voice_r.py
VOICE_MODEL_FILENAME = 'deepfake_detector.keras' # The saved model file for voice detection
VOICE_SAMPLE_RATE = 16000
VOICE_DURATION = 5  # seconds
VOICE_N_MELS = 128

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="Deepfake & Forgery Detector",
    layout="centered",
    initial_sidebar_state="auto" # Set to 'auto' to show the sidebar by default
)

st.title("🚨 Deepfake & Forgery Detection Suite")
st.markdown("Select a model, upload your file, and get an analysis.")

# --- Model Initialization Function ---
def initialize_models():
    """Initializes all models and clients, storing them in session_state."""
    # Roboflow Inference Client (for face_r.py)
    st.session_state.CLIENT = None
    try:
        from inference_sdk import InferenceHTTPClient
        st.session_state.CLIENT = InferenceHTTPClient(
            api_url=ROBOFLOW_API_URL,
            api_key=ROBOFLOW_API_KEY
        )
        st.toast("Roboflow Inference Client initialized.", icon="✅")
    except ImportError:
        st.toast("`inference-sdk` not found. Face Deepfake Detection will not work.", icon="⚠️")
    except Exception as e:
        st.toast(f"Error initializing Roboflow Inference Client: {e}. Ensure server is running.", icon="❌")

    # OpenCV Haar Cascade (for face_r.py)
    st.session_state.face_cascade = None
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if os.path.exists(cascade_path):
            st.session_state.face_cascade = cv2.CascadeClassifier(cascade_path)
            if st.session_state.face_cascade.empty():
                st.toast(f"Could not load Haar cascade from {cascade_path}.", icon="❌")
                st.session_state.face_cascade = None
            else:
                st.toast("OpenCV Haar cascade loaded.", icon="✅")
        else:
            st.toast(f"Haar cascade file not found at: {cascade_path}. Face detection will not work.", icon="❌")
    except Exception as e:
        st.toast(f"Error loading Haar cascade: {e}. Face detection will not work.", icon="❌")

    # Load Voice Deepfake Model (for voice_r.py)
    st.session_state.voice_model = None
    try:
        import tensorflow as tf
        import librosa
        if os.path.exists(VOICE_MODEL_FILENAME):
            st.session_state.voice_model = tf.keras.models.load_model(VOICE_MODEL_FILENAME)
            st.toast("Voice Deepfake Model loaded.", icon="✅")
        else:
            st.toast(f"Voice model file '{VOICE_MODEL_FILENAME}' not found. Voice Deepfake Detection will not work.", icon="❌")
    except ImportError:
        st.toast("`tensorflow` or `librosa` not found. Voice Deepfake Detection will not work.", icon="⚠️")
    except Exception as e:
        st.toast(f"Error loading voice model: {e}. Voice Deepfake Detection will not work.", icon="❌")

    st.session_state.models_initialized = True

# Run initialization only once per session
if 'models_initialized' not in st.session_state:
    initialize_models()

st.markdown("---") # Keep the separator if desired


# --- Helper Functions (from original scripts, adapted for Streamlit) ---

# --- FACE DEEPFAKE DETECTION (from face_r.py) ---
IMG_WIDTH = 256
IMG_HEIGHT = 256

def preprocess_frame(frame):
    """
    Detects faces, crops them, resizes, and prepares them for the Roboflow model.
    Returns a list of preprocessed face images (NumPy arrays).
    """
    # Access face_cascade from session_state
    face_cascade_local = st.session_state.get('face_cascade')
    if face_cascade_local is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_local.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

    preprocessed_faces = []
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
        preprocessed_faces.append(face)
    return preprocessed_faces

def predict_deepfake_on_video_roboflow_streamlit(file_path, client, model_ids, frame_interval=10, confidence_threshold=0.2):
    """
    Analyzes a video or image for deepfakes using an ensemble of Roboflow Inference API models.
    Adapted for Streamlit output.
    """
    # Access models from session_state
    client_local = st.session_state.get('CLIENT')
    face_cascade_local = st.session_state.get('face_cascade')

    if client_local is None or face_cascade_local is None or not model_ids:
        st.error("Face Deepfake Detection dependencies not met. Please ensure Roboflow server is running and Haar cascade is loaded.")
        return "Cannot perform detection due to uninitialized dependencies."

    output_messages = []
    
    # Check if it's an image or video
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith('image'):
        # Process as image
        try:
            frame = cv2.imread(file_path)
            if frame is None:
                return f"Error: Could not read image file {os.path.basename(file_path)}."
            
            preprocessed_faces = preprocess_frame(frame)
            if not preprocessed_faces:
                return "No faces detected in the image."

            output_messages.append(f"Analyzing image: {os.path.basename(file_path)}")
            
            num_parallel_model_calls = min(len(model_ids), os.cpu_count() or 1)
            output_messages.append(f"Using {num_parallel_model_calls} parallel workers for model inference.")

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel_model_calls) as executor:
                future_to_model_id = {
                    executor.submit(client_local.infer, preprocessed_faces, model_id=mid): mid 
                    for mid in model_ids
                }
                
                all_models_all_faces_confidences = []

                for future in concurrent.futures.as_completed(future_to_model_id):
                    model_id = future_to_model_id[future]
                    try:
                        batch_result = future.result() 
                        
                        if 'predictions' in batch_result:
                            if len(batch_result['predictions']) == len(preprocessed_faces):
                                model_face_confidences = []
                                for pred in batch_result['predictions']:
                                    if 'class' in pred and 'confidence' in pred:
                                        confidence = pred['confidence']
                                        prediction_class = pred['class']
                                        if confidence >= confidence_threshold:
                                            if prediction_class.lower() == 'fake':
                                                model_face_confidences.append(confidence)
                                            else:
                                                model_face_confidences.append(0.0)
                                        else:
                                            model_face_confidences.append(0.0) 
                                    else:
                                        model_face_confidences.append(0.0) 
                                all_models_all_faces_confidences.append(model_face_confidences)
                            else:
                                output_messages.append(f"  Model {model_id} predictions mismatch input face count. Expected {len(preprocessed_faces)}, got {len(batch_result['predictions'])}.")
                        else:
                            output_messages.append(f"  Model {model_id} returned no 'predictions' key for the batch.")

                    except Exception as api_e:
                        output_messages.append(f"  Error during Roboflow API inference for model {model_id} on batch: {api_e}")

                if all_models_all_faces_confidences and all(len(lst) == len(preprocessed_faces) for lst in all_models_all_faces_confidences):
                    ensemble_matrix = np.array(all_models_all_faces_confidences).T

                    ensemble_fake_predictions_count = 0
                    ensemble_real_predictions_count = 0
                    total_ensemble_fake_probability = 0
                    predictions_made_on_faces = 0

                    for face_idx in range(ensemble_matrix.shape[0]):
                        avg_fake_confidence_for_face = np.mean(ensemble_matrix[face_idx])
                        
                        total_ensemble_fake_probability += avg_fake_confidence_for_face
                        predictions_made_on_faces += 1

                        if avg_fake_confidence_for_face > 0.5:
                            ensemble_fake_predictions_count += 1
                        else:
                            ensemble_real_predictions_count += 1
                    
                    output_messages.append(f"  Analyzed {len(preprocessed_faces)} face(s). Ensemble Avg Fake Confidence: {total_ensemble_fake_probability / predictions_made_on_faces:.4f}")
                else:
                    output_messages.append(f"  Inconsistent prediction results for faces in batch. Skipping aggregation for this image.")
            
        except Exception as e:
            return f"An error occurred during image processing: {e}"

    elif mime_type and mime_type.startswith('video'):
        # Process as video
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return f"Error: Could not open video {os.path.basename(file_path)}. Please check the path and file integrity."

        frame_count = 0
        ensemble_fake_predictions_count = 0
        ensemble_real_predictions_count = 0
        total_ensemble_fake_probability = 0
        predictions_made_on_faces = 0

        output_messages.append(f"Analyzing video: {os.path.basename(file_path)} using Roboflow ensemble models: {', '.join(model_ids)}")
        output_messages.append(f"Processing every {frame_interval}th frame with confidence threshold {confidence_threshold*100:.0f}%.")

        num_parallel_model_calls = min(len(model_ids), os.cpu_count() or 1)
        output_messages.append(f"Using {num_parallel_model_calls} parallel workers for model inference.")

        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel_model_calls) as executor:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break # End of video

                if frame_count % frame_interval == 0:
                    preprocessed_faces = preprocess_frame(frame)
                    
                    if preprocessed_faces:
                        future_to_model_id = {
                            executor.submit(client_local.infer, preprocessed_faces, model_id=mid): mid 
                            for mid in model_ids
                        }
                        
                        all_models_all_faces_confidences = []

                        for future in concurrent.futures.as_completed(future_to_model_id):
                            model_id = future_to_model_id[future]
                            try:
                                batch_result = future.result() 
                                
                                if 'predictions' in batch_result:
                                    if len(batch_result['predictions']) == len(preprocessed_faces):
                                        model_face_confidences = []
                                        for pred in batch_result['predictions']:
                                            if 'class' in pred and 'confidence' in pred:
                                                confidence = pred['confidence']
                                                prediction_class = pred['class']
                                                if confidence >= confidence_threshold:
                                                    if prediction_class.lower() == 'fake':
                                                        model_face_confidences.append(confidence)
                                                    else:
                                                        model_face_confidences.append(0.0)
                                                else:
                                                    model_face_confidences.append(0.0) 
                                            else:
                                                model_face_confidences.append(0.0) 
                                        all_models_all_faces_confidences.append(model_face_confidences)

                            except Exception as api_e:
                                output_messages.append(f"  Frame {frame_count}: Error during Roboflow API inference for model {model_id} on batch: {api_e}")

                        if all_models_all_faces_confidences and all(len(lst) == len(preprocessed_faces) for lst in all_models_all_faces_confidences):
                            ensemble_matrix = np.array(all_models_all_faces_confidences).T

                            for face_idx in range(ensemble_matrix.shape[0]):
                                avg_fake_confidence_for_face = np.mean(ensemble_matrix[face_idx])
                                
                                total_ensemble_fake_probability += avg_fake_confidence_for_face
                                predictions_made_on_faces += 1

                                if avg_fake_confidence_for_face > 0.5:
                                    ensemble_fake_predictions_count += 1
                                else:
                                    ensemble_real_predictions_count += 1
                            
                frame_count += 1
                progress_bar.progress(min(100, int((frame_count / total_frames) * 100)))

        cap.release()
    else:
        return f"Unsupported file type for face detection: {mime_type}"

    results = {
        "frames_processed": frame_count if mime_type and mime_type.startswith('video') else 1,
        "faces_analyzed_by_ensemble": predictions_made_on_faces,
        "ensemble_fake_predictions": ensemble_fake_predictions_count,
        "ensemble_real_predictions": ensemble_real_predictions_count,
        "average_ensemble_fake_probability": total_ensemble_fake_probability / predictions_made_on_faces if predictions_made_on_faces > 0 else 0
    }
    
    output_messages.append("\n--- Deepfake Detection Results ---")
    for key, value in results.items():
        if isinstance(value, float):
            output_messages.append(f"{key}: {value:.4f}")
        else:
            output_messages.append(f"{key}: {value}")

    overall_deepfake_threshold = 0.67
    min_fake_faces_ratio = 0.2

    if predictions_made_on_faces > 0:
        actual_fake_faces_ratio = ensemble_fake_predictions_count / predictions_made_on_faces
        
        if results["average_ensemble_fake_probability"] >= overall_deepfake_threshold and actual_fake_faces_ratio >= min_fake_faces_ratio:
            output_messages.append(f"\nOverall Assessment: DEEPFAKE (Avg prob {results['average_ensemble_fake_probability']:.4f} >= {overall_deepfake_threshold:.2f} AND Fake faces ratio {actual_fake_faces_ratio:.2f} >= {min_fake_faces_ratio:.2f})")
        else:
            output_messages.append(f"\nOverall Assessment: REAL (Did not meet both deepfake criteria)")
    else:
        output_messages.append("\nOverall Assessment: CANNOT MAKE ASSESSMENT (No faces were analyzed.)")
    
    return "\n".join(output_messages)


# --- RECEIPT FORGERY DETECTION (from receipt.py) ---
def get_image_data_streamlit(file_content, mime_type):
    """
    Encodes an image or the first page of a PDF to a base64 string from bytes content.
    """
    if not file_content:
        return None, "Error: No file content provided."

    if mime_type == 'application/pdf':
        try:
            from pdf2image import convert_from_bytes
            images = convert_from_bytes(file_content.getvalue(), first_page=1, last_page=1)
            if not images:
                return None, "Error: Could not extract any pages from the PDF. Ensure Poppler is installed."
            
            image = images[0]
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return "image/png", encoded_string

        except ImportError:
            return None, "Error: `pdf2image` library not installed. Please install it."
        except Exception as e:
            return None, f"Error converting PDF: {e}. Ensure Poppler is installed and in your system's PATH."
            
    elif mime_type.startswith("image"):
        encoded_string = base64.b64encode(file_content.getvalue()).decode('utf-8')
        return mime_type, encoded_string
        
    else:
        return None, f"Error: File type '{mime_type}' is not a recognized image or PDF type."

def analyze_document_streamlit(file_content, mime_type):
    """
    Analyzes a document (image or PDF) for signs of forgery using the Gemini API.
    Adapted for Streamlit output.
    """
    if not GEMINI_API_KEY or "AIzaSyBh8f2N8VvKPJcsfH7k417xFnxlRyR8IpI" not in GEMINI_API_KEY:
        return "Error: Gemini API Key is not set correctly. Please replace the placeholder."

    image_mime_type, base64_image = get_image_data_streamlit(file_content, mime_type)
    if not image_mime_type:
        return base64_image # This will contain the error message

    prompt = dedent("""
        Act as a forensic document examiner. Analyze the following image of a receipt for any signs of digital manipulation, forgery, or being a "deepfake". Provide a detailed, point-by-point analysis focusing on subtle inconsistencies.

        **1. Font and Character Analysis:**
        - **Consistency:** Are all characters of the same type (e.g., all '0's) perfectly identical, or do they show natural print variations? Are there any minute differences in font weight, kerning, or baseline alignment for characters within the same word or line?
        - **Edge Quality:** Zoom in on character edges. Do they show signs of natural ink bleed and paper texture, or are they artificially sharp and clean, suggesting digital insertion?
        - **Placement:** Does any text appear to be misaligned with the rest of the line, or unnaturally positioned?

        **2. Background and Texture Analysis:**
        - **Inpainting Artifacts:** Scrutinize the blank paper background, especially around numbers and key text. Look for areas where the paper texture suddenly becomes blurry, smeared, or unnaturally uniform. This is a key sign that text has been digitally erased and the background "repaired" using an inpainting algorithm.
        - **Compression Anomalies (ELA):** Are there localized differences in JPEG compression levels? For example, does one number or word appear significantly noisier or blockier than the surrounding text, suggesting it was pasted from a different source image?
        - **Shadow and Lighting Consistency:** Is the lighting across the entire document uniform? Look for any text or blocks that cast unnatural shadows or lack the subtle shadows consistent with the rest of the document.

        **3. Structural and Layout Analysis:**
        - **Line Integrity:** Are printed lines (e.g., table borders, underlines) perfectly straight and consistent in thickness, or do they show slight waviness or breaks indicative of a real print? Check for any lines that appear digitally drawn.
        - **Alignment:** Do columns of numbers or text align perfectly, or is there a slight, natural-looking jitter?

        **4. Overall Conclusion:**
        - Based on the points above, provide a summary of your confidence level (e.g., High Confidence of Authenticity, Minor Anomalies Detected, Strong Indicators of Tampering).
        - Justify your conclusion by referencing the specific artifacts you found (or didn't find).
    """)

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": image_mime_type,
                            "data": base64_image
                        }
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(GEMINI_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if (result.get('candidates') and 
            result['candidates'][0].get('content') and 
            result['candidates'][0]['content'].get('parts')):
            
            analysis_text = result['candidates'][0]['content']['parts'][0]['text']
            return analysis_text
        else:
            return f"Error: Could not extract a valid analysis from the API response.\nFull Response: {result}"

    except requests.exceptions.RequestException as e:
        return f"An error occurred while calling the Gemini API: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# --- VOICE DEEPFAKE DETECTION (from voice_r.py) ---
def extract_features_streamlit(file_path):
    """Extracts a Mel spectrogram from a single audio file."""
    try:
        # Ensure librosa is imported inside the function if it might not be global
        import librosa
        audio, _ = librosa.load(file_path, sr=VOICE_SAMPLE_RATE, duration=VOICE_DURATION)
        
        if len(audio) < VOICE_DURATION * VOICE_SAMPLE_RATE:
            audio = np.pad(audio, (0, VOICE_DURATION * VOICE_SAMPLE_RATE - len(audio)), 'constant')
        else:
            audio = audio[:VOICE_DURATION * VOICE_SAMPLE_RATE]
        
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=VOICE_SAMPLE_RATE, n_mels=VOICE_N_MELS)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        return log_mel_spectrogram
    except ImportError:
        return "Error: `librosa` library not found. Please install it."
    except Exception as e:
        return f"Error processing audio file: {e}"

def verify_audio_streamlit(file_path):
    """
    Loads an audio file, preprocesses it, and predicts if it's real or fake.
    Adapted for Streamlit output.
    """
    # Access voice_model from session_state
    voice_model_local = st.session_state.get('voice_model')

    if voice_model_local is None:
        return "Voice Deepfake Detection model not loaded. Cannot perform this analysis."

    output_messages = []
    output_messages.append(f"Verifying audio file: {os.path.basename(file_path)}")
    
    features = extract_features_streamlit(file_path)
    
    if isinstance(features, str): # Check if it's an error message
        return features
    if features is None:
        return "Could not extract features from the audio file."
        
    features = features.reshape(1, features.shape[0], features.shape[1], 1)
    
    prediction_prob = voice_model_local.predict(features)[0][0]
    
    output_messages.append(f"Model's Raw Prediction Score: {prediction_prob:.4f}")
    if prediction_prob > 0.5:
        output_messages.append("Prediction: This audio is likely FAKE (spoofed).")
    else:
        output_messages.append("Prediction: This audio is likely REAL (bonafide).")
    
    return "\n".join(output_messages)


# --- Model Selection (in sidebar) ---
st.sidebar.header("Choose Your Analysis Model")
model_choice = st.sidebar.radio(
    "Select a model:",
    ("Face Deepfake Detection", "Receipt Forgery Detection", "Voice Deepfake Detection"),
    index=0,
    key="model_selector"
)

# Map display name to internal script name
model_map = {
    "Face Deepfake Detection": "face_r",
    "Receipt Forgery Detection": "receipt",
    "Voice Deepfake Detection": "voice_r"
}
selected_model_script = model_map[model_choice]

# File Uploader
st.header("Upload Your File")
allowed_extensions_dict = {
    "face_r": ["mp4", "avi", "mov", "mkv", "png", "jpg", "jpeg"],
    "receipt": ["png", "jpg", "jpeg", "pdf"],
    "voice_r": ["wav", "mp3", "flac", "m4a"]
}
allowed_extensions_for_current_model = allowed_extensions_dict[selected_model_script]

uploaded_file = st.file_uploader(
    f"Upload a file for {model_choice}",
    type=allowed_extensions_for_current_model,
    key="file_uploader"
)

analysis_button = st.button("Analyze File", key="analyze_button")

st.header("Analysis Results")
results_placeholder = st.empty() # Placeholder for results

if analysis_button and uploaded_file is not None:
    # Create a temporary file to save the uploaded content
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        mime_type = uploaded_file.type

        with st.spinner(f"Analyzing with {model_choice}... This might take a while for videos."):
            if selected_model_script == "face_r":
                # Access CLIENT and face_cascade from session_state for the function call
                result = predict_deepfake_on_video_roboflow_streamlit(
                    temp_file_path, st.session_state.get('CLIENT'), ROBOFLOW_MODEL_IDS,
                    frame_interval=10, confidence_threshold=0.8
                )
                results_placeholder.code(result)
            elif selected_model_script == "receipt":
                result = analyze_document_streamlit(uploaded_file, mime_type)
                results_placeholder.code(result)
            elif selected_model_script == "voice_r":
                # Access voice_model from session_state for the function call
                result = verify_audio_streamlit(temp_file_path)
                results_placeholder.code(result)
            else:
                results_placeholder.error("Invalid model selected.")
elif analysis_button and uploaded_file is None:
    st.warning("Please upload a file to analyze.")

st.markdown("---")
st.markdown(
    """
    **Note on Dependencies:**
    * **Face Deepfake Detection:** Requires a local Roboflow Inference Server running.
    * **Receipt Forgery Detection:** Requires Poppler to be installed on your system for PDF processing.
    * **Voice Deepfake Detection:** Requires the `deepfake_detector.keras` model file in the same directory.
    """
)
