import os
import cv2
import numpy as np

def preprocess_images(data_dir):
    images = []
    labels = []

    for label in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, label)
        if os.path.isdir(person_dir):
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (100, 100))
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)


if __name__ == "__main__":
    images, labels = preprocess_images('.../data/lfw/')