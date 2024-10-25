import os
import cv2
import numpy as np

def preprocess_images(data_dir):
    images = []
    labels = []
    label_map = {}  # To store the mapping of labels to integers
    label_id = 0    # Start label IDs from 0

    # Loop through each folder in the data directory
    for person in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person)
        if os.path.isdir(person_dir):
            if person not in label_map:
                label_map[person] = label_id
                label_id += 1

            # Loop through each image in the person directory
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                img = cv2.imread(image_path)
                img = cv2.resize(img, (100, 100))  # Resize image to match model input
                images.append(img)
                labels.append(label_map[person])  # Use the integer label

    # Convert lists to numpy arrays
    images = np.array(images, dtype='float32') / 255.0  # Normalize the images
    labels = np.array(labels, dtype='int')  # Ensure labels are integers

    return images, labels
