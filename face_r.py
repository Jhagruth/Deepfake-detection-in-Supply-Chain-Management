import cv2
import numpy as np
import os
from inference_sdk import InferenceHTTPClient
import concurrent.futures # For parallelizing API calls
import math # For math.ceil

# --- Roboflow API Configuration ---
# IMPORTANT: Use your actual Roboflow API URL and Key here.
# When running locally, point to your local Inference Server (e.g., http://localhost:9001).
ROBOFLOW_API_URL = "http://localhost:9001" # <<<--- Ensure this points to your local Roboflow Inference Server
ROBOFLOW_API_KEY = "zWV15F4167MkKAlS4nVh"
# List of Roboflow model IDs to use for ensemble prediction
ROBOFLOW_MODEL_IDS = [
    "deepfake-yryoa/1",
    "deepfake-detection-v5wyo/1",
    "deepfake-apkbz/1"
]

# Initialize the Roboflow Inference Client globally
try:
    CLIENT = InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=ROBOFLOW_API_KEY
    )
    print("Roboflow Inference Client initialized successfully.")
    print(f"Using {len(ROBOFLOW_MODEL_IDS)} Roboflow models for ensemble prediction.")
except Exception as e:
    print(f"Error initializing Roboflow Inference Client: {e}")
    print("Please ensure your local Roboflow Inference Server is running and accessible.")
    print("You can start it using: 'inference server start --dev'")
    CLIENT = None # Set to None to indicate failure

# --- Face Detection and Preprocessing (using OpenCV) ---
face_cascade = None # Initialize as None
try:
    # Attempt to load the Haar Cascade file.
    # The path `cv2.data.haarcascades` points to where OpenCV expects to find its pre-trained cascades.
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    if not os.path.exists(cascade_path):
        print(f"CRITICAL ERROR: Haar cascade file not found at: {cascade_path}")
        print("Please ensure OpenCV is correctly installed and its data files are accessible.")
        print("You might need to manually locate 'haarcascade_frontalface_default.xml' and place it where your script can find it, or ensure your OpenCV installation is complete.")
    else:
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print(f"CRITICAL ERROR: Could not load Haar cascade file from {cascade_path} even though it exists.")
            print("The file might be corrupted or unreadable.")
            face_cascade = None
except Exception as e:
    print(f"Error during Haar cascade initialization: {e}")
    print("Face detection will not work. Please check your OpenCV installation.")
    face_cascade = None

IMG_WIDTH = 256
IMG_HEIGHT = 256

def preprocess_frame(frame):
    """
    Detects faces, crops them, resizes, and prepares them for the Roboflow model.
    Returns a list of preprocessed face images (NumPy arrays).
    """
    if face_cascade is None:
        # If cascade failed to load, cannot perform face detection.
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Adjust parameters (scaleFactor, minNeighbors) as needed for your use case.
    # minSize can help filter out very small, potentially false-positive faces.
    # You might need to adjust these for different video qualities or face sizes.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

    preprocessed_faces = []
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
        preprocessed_faces.append(face)
    return preprocessed_faces

# --- Prediction Function using Roboflow API ---

def predict_deepfake_on_video_roboflow(video_path, client, model_ids, frame_interval=10, confidence_threshold=0.2):
    """
    Analyzes a video for deepfakes using an ensemble of Roboflow Inference API models.
    API calls to multiple models for a single frame's faces are now batched and parallelized.

    Args:
        video_path (str): Path to the input video file.
        client (InferenceHTTPClient): The initialized Roboflow Inference client.
        model_ids (list): A list of IDs of the deepfake detection models on Roboflow.
        frame_interval (int): Process every Nth frame to speed up analysis.
        confidence_threshold (float): Minimum confidence for a prediction to be considered.
                                      Predictions below this threshold will be ignored.

    Returns:
        dict: A dictionary with counts of real/fake predictions and average fake probability.
              Returns an empty dict if the video cannot be opened or client is not initialized.
    """
    if client is None:
        print("Roboflow Inference Client is not initialized. Cannot perform prediction.")
        return {}
    if not model_ids:
        print("No Roboflow model IDs provided. Cannot perform prediction.")
        return {}
    if face_cascade is None:
        print("Face detection is not initialized. Cannot perform prediction.")
        return {}


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}. Please check the path and file integrity.")
        return {}

    frame_count = 0
    ensemble_fake_predictions_count = 0
    ensemble_real_predictions_count = 0
    total_ensemble_fake_probability = 0
    predictions_made_on_faces = 0

    print(f"Analyzing video: {video_path} using Roboflow ensemble models: {', '.join(model_ids)}")
    print(f"Processing every {frame_interval}th frame with confidence threshold {confidence_threshold*100:.0f}%.")

    num_parallel_model_calls = min(len(model_ids), os.cpu_count() or 1)
    print(f"Using {num_parallel_model_calls} parallel workers for model inference.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel_model_calls) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video

            if frame_count % frame_interval == 0:
                preprocessed_faces = preprocess_frame(frame)
                
                if preprocessed_faces: # Only process if faces are detected in the current frame
                    # Submit inference tasks for each model in parallel for the *batch* of faces
                    # Removed 'confidence' keyword argument from infer call
                    future_to_model_id = {
                        executor.submit(client.infer, preprocessed_faces, model_id=mid): mid 
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
                                            # Filter by confidence_threshold after receiving prediction
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
                                    print(f"  Frame {frame_count}: Model {model_id} predictions mismatch input face count. Expected {len(preprocessed_faces)}, got {len(batch_result['predictions'])}.")
                            else:
                                print(f"  Frame {frame_count}: Model {model_id} returned no 'predictions' key for the batch.")

                        except Exception as api_e:
                            print(f"  Frame {frame_count}: Error during Roboflow API inference for model {model_id} on batch: {api_e}")

                    if all_models_all_faces_confidences:
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
                            
                            print(f"  Frame {frame_count}: Analyzed {len(preprocessed_faces)} face(s). Current Ensemble Avg Fake Confidence: {total_ensemble_fake_probability / predictions_made_on_faces:.4f}")
                        else:
                            print(f"  Frame {frame_count}: Inconsistent prediction results for faces in batch. Skipping aggregation for this frame.")
                    else:
                        print(f"  Frame {frame_count}: No valid predictions from any model for detected faces in this frame.")
                else:
                    print(f"  Frame {frame_count}: No faces detected in this frame.") # Explicit message when no faces are found

            frame_count += 1

    cap.release()

    results = {
        "frames_processed": frame_count,
        "faces_analyzed_by_ensemble": predictions_made_on_faces,
        "ensemble_fake_predictions": ensemble_fake_predictions_count,
        "ensemble_real_predictions": ensemble_real_predictions_count,
        "average_ensemble_fake_probability": total_ensemble_fake_probability / predictions_made_on_faces if predictions_made_on_faces > 0 else 0
    }
    return results

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Deepfake Detection using Roboflow Hosted Ensemble Models ---")

    # --- Prediction Example Usage ---
    # Replace 'path/to/your/video_for_prediction.mp4' with the actual path to a video file on your system.
    video_file_to_analyze = '/Users/jhagruth/Downloads/Hackathon/my_deepfake_images/test/Fake/image_00341.png' # <<<--- CHANGE THIS TO YOUR VIDEO FILE PATH

    # Check if the Roboflow client and face cascade are initialized before proceeding
    if CLIENT is None:
        print("\nCannot proceed: Roboflow Inference Client was not initialized. Please ensure your local Roboflow Inference Server is running and accessible.")
    elif face_cascade is None:
        print("\nCannot proceed: Face detection cascade was not loaded. Please check your OpenCV installation and the Haar cascade file.")
    elif not os.path.exists(video_file_to_analyze):
        print(f"\nVideo file not found at '{video_file_to_analyze}'.")
        print("Please provide a valid path to a video file to run the detection.")
        print("Example: 'C:/Users/YourUser/Videos/test_video.mp4' or '/home/user/videos/test.mov'")
    else:
        print(f"\nStarting deepfake detection on: {video_file_to_analyze}")
        
        # Adjust these parameters for speed vs. accuracy tradeoff:
        # frame_interval: Increase to skip more frames (faster, less granular).
        # confidence_threshold: This filters individual face predictions based on their confidence.
        # overall_deepfake_threshold: Adjust this to control the final video classification based on average fake probability.
        # min_fake_faces_ratio: NEW! Adjust this to require a minimum proportion of faces to be fake.
        detection_results = predict_deepfake_on_video_roboflow(
            video_file_to_analyze,
            CLIENT,
            ROBOFLOW_MODEL_IDS,
            frame_interval=10, # Example: 1 (every frame), 5 (every 5th frame), 10 (every 10th frame)
            confidence_threshold=0.8 # Minimum confidence for an individual face prediction (0.0 to 1.0)
        )

        if detection_results:
            print("\n--- Deepfake Detection Results ---")
            for key, value in detection_results.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")

            # --- Updated Logic for Overall Video Assessment with min_fake_faces_ratio ---
            faces_analyzed = detection_results.get("faces_analyzed_by_ensemble", 0)
            avg_ensemble_prob = detection_results.get("average_ensemble_fake_probability", 0)
            fake_predictions_count = detection_results.get("ensemble_fake_predictions", 0)
            
            # Define the threshold for overall video assessment based on average probability
            # Increase this value if you want to be more conservative in flagging videos as deepfakes.
            overall_deepfake_threshold = 0.67 # Example: 0.55, 0.6, 0.7

            # Define the minimum ratio of fake faces required for overall deepfake classification
            # Increase this value (e.g., to 0.3, 0.4, 0.5) to reduce false positives.
            # This means a higher proportion of detected faces must be classified as fake.
            min_fake_faces_ratio = 0.2 # Example: 0.2 (20%), 0.3 (30%), 0.5 (50%)

            if faces_analyzed > 0:
                # Calculate the actual ratio of fake predictions
                actual_fake_faces_ratio = fake_predictions_count / faces_analyzed
                
                if avg_ensemble_prob >= overall_deepfake_threshold and actual_fake_faces_ratio >= min_fake_faces_ratio:
                    print(f"\nOverall Video Assessment: DEEPFAKE (Avg prob {avg_ensemble_prob:.4f} >= {overall_deepfake_threshold:.2f} AND Fake faces ratio {actual_fake_faces_ratio:.2f} >= {min_fake_faces_ratio:.2f})")
                else:
                    print(f"\nOverall Video Assessment: REAL (Did not meet both deepfake criteria)")
            else:
                print("\nOverall Video Assessment: CANNOT MAKE ASSESSMENT (No faces were analyzed in the video.)")
        else:
            print("\nCould not perform detection. Check video path and permissions, or Roboflow client initialization.")
