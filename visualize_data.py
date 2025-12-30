import cv2
import matplotlib.pyplot as plt
import os
import random

# --- CONFIGURATION ---
# IMPORTANT: Update these paths to your folders!
image_folder = 'D:/Production/NEU-DET/train/images'
label_folder = 'D:/Production/NEU-DET/train/annotations'

# These are the names for the 6 defect classes in the NEU dataset
class_names = [
    'crazing', 'inclusion', 'patches', 
    'pitted_surface', 'rolled-in_scale', 'scratches'
]
# --- END CONFIGURATION ---

# --- Automatically select a random image ---
# Get a list of all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
# Pick one random filename
random_image_name = random.choice(image_files)

# Create the full paths for the image and its corresponding label
image_path = os.path.join(image_folder, random_image_name)
# Change the file extension from .jpg to .txt for the label file
base_name = os.path.splitext(random_image_name)[0]
label_path = os.path.join(label_folder, base_name + '.txt')

print(f"Displaying random image: {random_image_name}")
# ---------------------------------------------


# Read the image using OpenCV
image = cv2.imread(image_path)
# OpenCV reads images in BGR format, so we convert it to RGB for Matplotlib
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, _ = image.shape

# Read the label file
if os.path.exists(label_path):
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:])
            
            x_center = x_center_norm * w
            y_center = y_center_norm * h
            box_width = width_norm * w
            box_height = height_norm * h
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))

            label_name = class_names[class_id]
            cv2.rectangle(image, (x_min, y_min), (int(x_min + box_width), int(y_min + box_height)), (0, 255, 0), 2)
            cv2.putText(image, label_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
else:
    print(f"Warning: No label file found for {random_image_name}")

# Display the image with the bounding box
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.title(f"Image: {random_image_name}")
plt.axis('off')
plt.show()