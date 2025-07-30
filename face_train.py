import os
from datasets import load_dataset, DatasetDict
from PIL import Image # Used for saving images

def download_and_save_deepfake_images(dataset_name: str, output_root_dir: str):
    """
    Downloads the specified Hugging Face dataset and saves its images
    to a local directory structure organized by split and label.

    Args:
        dataset_name (str): The name of the dataset on Hugging Face (e.g., "JamieWithofs/Deepfake-and-real-images").
        output_root_dir (str): The root directory where the images will be saved.
                               E.g., if "my_deepfake_images", then images will go into
                               "my_deepfake_images/train/real/", "my_deepfake_images/validation/fake/", etc.
    """
    print(f"Loading dataset: '{dataset_name}' from Hugging Face...")
    try:
        ds = load_dataset(dataset_name)
        print("Dataset loaded successfully!")
        print(f"Dataset structure: {ds}")
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        print("Please ensure the dataset name is correct and you have an internet connection.")
        return

    # Ensure the output root directory exists
    os.makedirs(output_root_dir, exist_ok=True)

    # The dataset is a DatasetDict, so we iterate through its splits (e.g., 'train', 'validation', 'test')
    if isinstance(ds, DatasetDict):
        for split_name, dataset_split in ds.items():
            print(f"\nProcessing split: '{split_name}' (Total images: {len(dataset_split)})")
            
            # Check if the split contains an 'image' feature and a 'label' feature
            if 'image' not in dataset_split.features:
                print(f"  Skipping split '{split_name}': No 'image' feature found.")
                continue
            if 'label' not in dataset_split.features:
                print(f"  Skipping split '{split_name}': No 'label' feature found. Cannot organize by real/fake.")
                continue

            # Create split directory (e.g., "my_deepfake_images/train")
            split_dir_path = os.path.join(output_root_dir, split_name)
            os.makedirs(split_dir_path, exist_ok=True)

            # Get label names from the dataset features (e.g., 0 -> 'fake', 1 -> 'real')
            # This assumes the 'label' feature is a ClassLabel
            label_names = dataset_split.features['label'].names
            print(f"  Detected labels: {label_names}")

            for i, sample in enumerate(dataset_split):
                try:
                    image = sample['image'] # Get the PIL Image object
                    label_id = sample['label'] # Get the label ID (e.g., 0, 1)
                    label_name = label_names[label_id] # Convert ID to name (e.g., 'fake', 'real')

                    # Create label subdirectory (e.g., "my_deepfake_images/train/real")
                    label_dir_path = os.path.join(split_dir_path, label_name)
                    os.makedirs(label_dir_path, exist_ok=True)

                    # Define image filename
                    image_filename = f"image_{i:05d}.png" # Padded with zeros for sorting
                    image_path = os.path.join(label_dir_path, image_filename)

                    image.save(image_path)
                    if (i + 1) % 100 == 0:
                        print(f"  Saved {i + 1} images from '{split_name}' split...")
                except Exception as e:
                    print(f"  Error saving image {i} from split '{split_name}': {e}")
            print(f"Finished saving all images from split '{split_name}'.")
    else:
        print(f"Dataset '{dataset_name}' is not a DatasetDict. Cannot process splits.")
        print("Please check the dataset structure on Hugging Face Hub.")

    print(f"\nAll images from '{dataset_name}' have been processed and saved to '{output_root_dir}'.")

if __name__ == "__main__":
    # --- Configuration ---
    # The name of the dataset to download
    target_dataset_name = "JamieWithofs/Deepfake-and-real-images"

    # The local directory where you want to save the images.
    # This script will create subdirectories like:
    # my_deepfake_images/
    # ├── train/
    # │   ├── real/
    # │   └── fake/
    # ├── validation/
    # │   ├── real/
    # │   └── fake/
    # └── test/
    #     ├── real/
    #     └── fake/
    output_directory = "my_deepfake_images" # <<<--- CHANGE THIS TO YOUR DESIRED DOWNLOAD PATH

    # --- Execute Download and Save ---
    download_and_save_deepfake_images(target_dataset_name, output_directory)

    print(f"\nDataset images are now available in the '{output_directory}' directory.")
    print("You can use these images for local training or further analysis.")

