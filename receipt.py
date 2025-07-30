import os
import base64
import requests
import argparse
import mimetypes
from textwrap import dedent
from io import BytesIO

# --- New Dependency Check ---
try:
    from pdf2image import convert_from_path
except ImportError:
    print("Error: The 'pdf2image' library is not installed.")
    print("Please install it using: pip install pdf2image")
    print("You also need to install Poppler. For macOS, use: brew install poppler")
    exit()

# --- Configuration ---
# WARNING: Hardcoding API keys is insecure. It's better to use an environment variable.
API_KEY = "AIzaSyBh8f2N8VvKPJcsfH7k417xFnxlRyR8IpI" 
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

def get_image_data(file_path):
    """
    Encodes an image or the first page of a PDF to a base64 string.

    Args:
        file_path (str): The path to the file.

    Returns:
        tuple[str, str] | None: A tuple containing the mime type and the base64 encoded string,
                                or None if the file is not found or is an unsupported type.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return None

    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        print(f"Error: Could not determine file type for '{file_path}'.")
        return None

    # Handle PDF files
    if mime_type == 'application/pdf':
        print("PDF file detected. Converting the first page to an image...")
        try:
            images = convert_from_path(file_path, first_page=1, last_page=1)
            if not images:
                print("Error: Could not extract any pages from the PDF.")
                return None
            
            image = images[0]
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return "image/png", encoded_string

        except Exception as e:
            print(f"Error converting PDF: {e}")
            print("Please ensure Poppler is installed and in your system's PATH.")
            return None
            
    # Handle standard image files
    elif mime_type.startswith("image"):
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return mime_type, encoded_string
        
    else:
        print(f"Error: File '{file_path}' is not a recognized image or PDF type.")
        return None

def analyze_document(file_path):
    """
    Analyzes a document (image or PDF) for signs of forgery using the Gemini API.

    Args:
        file_path (str): The path to the document file.
    """
    if not API_KEY or "AIzaSyBh8f2N8VvKPJcsfH7k417xFnxlRyR8IpI" not in API_KEY:
        print("Error: API_KEY is not set correctly in the script.")
        print("Please replace the placeholder with your actual Google AI Studio API key.")
        return

    print(f"Analyzing '{file_path}'...")

    image_data = get_image_data(file_path)
    if not image_data:
        return

    mime_type, base64_image = image_data

    # --- IMPROVED PROMPT ---
    # This prompt is much more detailed and guides the model to act as a forensic expert.
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

    # Construct the request payload
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64_image
                        }
                    }
                ]
            }
        ]
    }

    # Make the API call
    try:
        print("Sending request to the Gemini API...")
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if (result.get('candidates') and 
            result['candidates'][0].get('content') and 
            result['candidates'][0]['content'].get('parts')):
            
            analysis_text = result['candidates'][0]['content']['parts'][0]['text']
            print("\n--- Analysis Result ---")
            print(analysis_text)
            print("-----------------------")
        else:
            print("\nError: Could not extract a valid analysis from the API response.")
            print("Full Response:", result)

    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred while calling the API: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a receipt/bill image or PDF for signs of digital forgery.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="The full path to the image or PDF file you want to analyze."
    )
    
    parser.epilog = dedent('''
    --- How to Run ---
    1. Make sure you have Python and the required libraries installed:
       pip install requests pdf2image
    2. You also need to install the Poppler utility, which is used by pdf2image.
       - For macOS (using Homebrew):
         brew install poppler
       - For Debian/Ubuntu:
         sudo apt-get install poppler-utils
       - For Windows:
         Download Poppler, extract it, and add the 'bin' folder to your system's PATH.
    3. Run the script from your terminal:
       python your_script_name.py /path/to/your/document.pdf
    ''')

    args = parser.parse_args()
    analyze_document(args.file_path)
