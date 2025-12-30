import xml.etree.ElementTree as ET
import os

# --- CONFIGURATION ---
# List of the annotation folders you want to process
ANNOTATION_FOLDERS = [
    'D:/Production/NEU-DET/train/annotations',
    'D:/Production/NEU-DET/validation/annotations'
]

# Path where the new folders with .txt label files will be saved
# We will create 'train/labels' and 'validation/labels' inside this.
TARGET_BASE_DIR = 'D:/Production/NEU-DET'

# The class names in the correct order for their ID (0, 1, 2, etc.)
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
# --- END CONFIGURATION ---

def convert_all_xml_to_yolo():
    """
    Converts all Pascal VOC (.xml) files from the specified folders
    into YOLO (.txt) format in a new 'labels' directory.
    """
    print("Starting XML to YOLO conversion...")

    for source_folder in ANNOTATION_FOLDERS:
        if not os.path.isdir(source_folder):
            print(f"Warning: Source folder not found, skipping: {source_folder}")
            continue

        # Determine if this is a 'train' or 'validation' folder
        split_name = os.path.basename(os.path.dirname(source_folder)) # e.g., 'train' or 'validation'
        
        # Create the corresponding target directory (e.g., 'D:/Production/NEU-DET/train/labels')
        target_labels_dir = os.path.join(TARGET_BASE_DIR, split_name, 'labels')
        os.makedirs(target_labels_dir, exist_ok=True)
        
        print(f"\nProcessing folder: '{source_folder}'")
        print(f"Saving .txt files to: '{target_labels_dir}'")

        for xml_file in os.listdir(source_folder):
            if not xml_file.endswith('.xml'):
                continue

            # Parse the XML file
            tree = ET.parse(os.path.join(source_folder, xml_file))
            root = tree.getroot()

            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)

            yolo_lines = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in CLASS_NAMES:
                    continue
                class_id = CLASS_NAMES.index(class_name)

                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                # Convert to YOLO format
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Save the new .txt file in the new 'labels' directory
            txt_filename = os.path.splitext(xml_file)[0] + '.txt'
            with open(os.path.join(target_labels_dir, txt_filename), 'w') as f:
                f.write('\n'.join(yolo_lines))
    
    print("\nConversion complete! New 'labels' folders have been created with .txt files.")

if __name__ == '__main__':
    convert_all_xml_to_yolo()