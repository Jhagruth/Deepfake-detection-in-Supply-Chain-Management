import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm
import numpy as np

# --- New Dependency Check ---
try:
    from datasets import load_dataset
except ImportError:
    print("Error: The 'datasets' library is not installed.")
    print("Please install it using: pip install datasets")
    exit()

try:
    import pytesseract
except ImportError:
    print("Error: The 'pytesseract' library is not installed.")
    print("Please install it using: pip install pytesseract")
    print("You also need to install the Tesseract-OCR engine itself.")
    print("For macOS: brew install tesseract")
    print("For Debian/Ubuntu: sudo apt-get install tesseract-ocr")
    exit()


# --- Configuration ---
OUTPUT_DIR = './receipt_dataset/train/fake'
DATASET_NAME = "mychen76/invoices-and-receipts_ocr_v1"


# --- Forgery Functions ---

def paste_random_block(img):
    """
    Cuts a random rectangular block from the image and pastes it at another
    random location, simulating a crude copy-paste forgery.
    """
    try:
        img_w, img_h = img.size
        block_w = random.randint(int(img_w * 0.1), int(img_w * 0.4))
        block_h = random.randint(int(img_h * 0.1), int(img_h * 0.4))
        src_x = random.randint(0, img_w - block_w)
        src_y = random.randint(0, img_h - block_h)
        dst_x = random.randint(0, img_w - block_w)
        dst_y = random.randint(0, img_h - block_h)
        box = (src_x, src_y, src_x + block_w, src_y + block_h)
        region = img.crop(box)
        if random.random() > 0.5:
            region = region.rotate(random.randint(-5, 5), expand=True, fillcolor=(255,255,255))
        img.paste(region, (dst_x, dst_y))
        return img
    except Exception as e:
        print(f"  - Could not apply paste_random_block: {e}")
        return img


def add_random_text(img):
    """
    Adds a random string of text at a random location on the image.
    This simulates someone adding fraudulent text.
    """
    try:
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size
        try:
            font = ImageFont.truetype("arial.ttf", size=random.randint(20, 45))
        except IOError:
            font = ImageFont.load_default()
        random_texts = [f"${random.randint(10, 500)}.{random.randint(10,99)}", f"ID: {random.randint(1000, 9999)}", "REFUNDED", "PAID", "VOID"]
        text = random.choice(random_texts)
        left, top, right, bottom = font.getbbox(text)
        text_w = right - left
        text_h = bottom - top
        x = random.randint(0, max(0, img_w - text_w))
        y = random.randint(0, max(0, img_h - text_h))
        text_color = random.randint(20, 80)
        draw.text((x, y), text, fill=(text_color, text_color, text_color), font=font)
        return img
    except Exception as e:
        print(f"  - Could not apply add_random_text: {e}")
        return img

def add_blotch(img):
    """
    Adds a semi-transparent blotch or stain to the image to simulate
    obscuring information.
    """
    try:
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        img_w, img_h = img.size
        x0 = random.randint(0, int(img_w * 0.8))
        y0 = random.randint(0, int(img_h * 0.8))
        x1 = x0 + random.randint(int(img_w * 0.1), int(img_w * 0.3))
        y1 = y0 + random.randint(int(img_h * 0.1), int(img_h * 0.3))
        blotch_color = (101, 67, 33, random.randint(70, 120))
        draw.ellipse((x0, y0, x1, y1), fill=blotch_color)
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=random.randint(5, 15)))
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        return img
    except Exception as e:
        print(f"  - Could not apply add_blotch: {e}")
        return img

def subtle_text_alteration(img):
    """
    **NEW: More realistic forgery using OCR.**
    Finds text containing numbers, erases it, and replaces it with slightly
    altered text to simulate a subtle forgery.
    """
    try:
        # Use pytesseract to get detailed data about text on the image
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        # Find all words that contain at least one digit
        numeric_blocks = []
        for i in range(len(data['text'])):
            # Filter for reasonably confident and non-empty text blocks
            if int(data['conf'][i]) > 40 and data['text'][i].strip() != '' and any(char.isdigit() for char in data['text'][i]):
                numeric_blocks.append(i)
        
        if not numeric_blocks:
            # If no numbers found, fall back to another forgery type
            return add_random_text(img)

        # Choose a random numeric block to alter
        block_index = random.choice(numeric_blocks)
        
        text = data['text'][block_index]
        x, y, w, h = data['left'][block_index], data['top'][block_index], data['width'][block_index], data['height'][block_index]

        # --- Erase the original text ---
        # Get the average color from the immediate background of the text
        background_box = (max(0, x-5), max(0, y-5), min(img.width, x+w+5), min(img.height, y+h+5))
        background = img.crop(background_box)
        avg_color = tuple(np.array(background).mean(axis=(0,1)).astype(int))
        
        draw = ImageDraw.Draw(img)
        # Draw a rectangle over the text with the average background color
        draw.rectangle([x, y, x + w, y + h], fill=avg_color, outline=avg_color)

        # --- Create and write the new, altered text ---
        original_digits = "0123456789"
        new_text = ""
        altered = False
        # Replace one digit in the text with a different random digit
        for char in text:
            if char.isdigit() and not altered:
                new_digit = random.choice(original_digits.replace(char, ''))
                new_text += new_digit
                altered = True
            else:
                new_text += char
        
        # If for some reason no digit was altered, just change the last one
        if not altered and len(new_text) > 0 and new_text[-1].isdigit():
             new_text = new_text[:-1] + random.choice(original_digits.replace(new_text[-1], ''))

        # Use a font size that is similar to the original text height
        font_size = h
        try:
            font = ImageFont.truetype("arial.ttf", size=font_size)
        except IOError:
            font = ImageFont.load_default()
        
        # Write the new text
        draw.text((x, y), new_text, fill=(10, 10, 10), font=font)
        
        return img

    except Exception as e:
        print(f"  - Could not apply subtle_text_alteration: {e}")
        # If OCR fails, fall back to a simpler method
        return paste_random_block(img)


# --- Main Script ---

def generate_fakes():
    """
    Main function to load a dataset from Hugging Face, apply random forgeries,
    and save them to the output directory.
    """
    if not os.path.exists(OUTPUT_DIR):
        print(f"Output directory not found. Creating it now: '{OUTPUT_DIR}'")
        os.makedirs(OUTPUT_DIR)

    print(f"Loading dataset '{DATASET_NAME}' from Hugging Face...")
    print("This may take a few minutes the first time as it downloads the data.")
    try:
        ds = load_dataset(DATASET_NAME)
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        return

    train_dataset = ds['train']
    print(f"Found {len(train_dataset)} images in the 'train' split. Starting forgery generation...")

    # Add the new subtle forgery function to the list of choices
    forgery_functions = [paste_random_block, add_random_text, add_blotch, subtle_text_alteration, subtle_text_alteration]

    # --- FIX: Start enumeration from 1 for human-readable filenames ---
    for i, example in enumerate(tqdm(train_dataset, desc="Generating Fakes"), start=1):
        try:
            img = example['image'].convert('RGB')
            output_path = os.path.join(OUTPUT_DIR, f"fake_receipt_{i}.jpg")
            chosen_forgery = random.choice(forgery_functions)
            fake_img = chosen_forgery(img)
            fake_img.save(output_path, 'JPEG', quality=random.randint(85, 95))
        except KeyError:
            print(f"\nSkipping item {i} because it does not have an 'image' key.")
        except Exception as e:
            print(f"\nCould not process item {i}. Reason: {e}")

    print("\nForgery generation complete!")
    print(f"Fake receipts have been saved in: '{OUTPUT_DIR}'")


if __name__ == '__main__':
    # --- How to Run ---
    # 1. Make sure you have the required Python libraries:
    #    pip install Pillow tqdm datasets pytesseract numpy
    #
    # 2. You MUST also install the Tesseract-OCR engine on your system.
    #    - For macOS (using Homebrew):
    #      brew install tesseract
    #    - For Debian/Ubuntu Linux:
    #      sudo apt-get update
    #      sudo apt-get install tesseract-ocr
    #    - For Windows:
    #      Download and run the installer from https://github.com/UB-Mannheim/tesseract/wiki
    #
    # 3. Run this script from your terminal:
    #    python your_script_name.py
    
    generate_fakes()
