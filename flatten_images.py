import os
import shutil

# --- CONFIGURATION ---
# The path to your main dataset folder that contains 'train' and 'validation'
BASE_DIR = 'D:/Production/NEU-DET'
# --- END CONFIGURATION ---

def flatten_image_folders():
    """
    Moves image files from class subfolders up to the parent 'images' folder.
    This script now IGNORES the 'annotations' folder, as it is already correct.
    """
    print(f"Starting to flatten the image directory structure in '{BASE_DIR}'...")
    
    # We need to process both the training and validation sets
    for split_folder in ['train', 'validation']:
        # The path to the images folder (e.g., 'D:/Production/NEU-DET/train/images/')
        images_root_path = os.path.join(BASE_DIR, split_folder, 'images')
        
        if not os.path.exists(images_root_path):
            print(f"Warning: Folder not found, skipping: {images_root_path}")
            continue

        print(f"\nProcessing: {images_root_path}")
        
        # Find all the class subfolders (e.g., 'crazing', 'patches')
        class_subfolders = [d for d in os.listdir(images_root_path) if os.path.isdir(os.path.join(images_root_path, d))]

        if not class_subfolders:
            print(" -> No subfolders found. The 'images' folder might already be flat.")
            continue

        for class_folder in class_subfolders:
            source_folder_path = os.path.join(images_root_path, class_folder)
            
            # Move every file from the subfolder to the parent 'images' folder
            for filename in os.listdir(source_folder_path):
                source_file = os.path.join(source_folder_path, filename)
                destination_file = os.path.join(images_root_path, filename)
                shutil.move(source_file, destination_file)
            
            # After moving all files, remove the now-empty class folder
            os.rmdir(source_folder_path)
            print(f" -> Moved images from '{class_folder}' and removed the empty folder.")

    print("\nImage folder flattening complete!")
    print("Your dataset should now be fully compatible with YOLO.")


if __name__ == '__main__':
    flatten_image_folders()