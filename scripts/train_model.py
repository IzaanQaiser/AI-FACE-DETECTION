import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from preprocess import preprocess_images

def create_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_data(data_dir):
    # Load processed images and labels
    images, labels = preprocess_images(data_dir)
    return images, labels

if __name__ == "__main__":
    # Define the data directory
    data_dir = 'data/lfw/'  # Adjust the path as necessary

    # Load data
    X, y = load_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = create_model(len(set(y)))
    history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Save the trained model
    model.save('../models/face_detection_model.h5')
