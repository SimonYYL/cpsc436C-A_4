import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import argparse
import os
import pickle
import numpy as np

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def load_cifar10_data(data_dir):
    train_batches = []
    train_labels = []
    for batch_idx in range(1, 6):
        batch_file = os.path.join(data_dir, f"data_batch_{batch_idx}")
        with open(batch_file, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        train_batches.append(batch[b"data"])
        train_labels.extend(batch[b"labels"])

    x_train = np.vstack(train_batches).reshape(-1, 32, 32, 3) / 255.0
    y_train = np.array(train_labels)

    test_file = os.path.join(data_dir, "test_batch")
    with open(test_file, "rb") as f:
        test_batch = pickle.load(f, encoding="bytes")
    x_test = test_batch[b"data"].reshape(-1, 32, 32, 3) / 255.0
    y_test = np.array(test_batch[b"labels"])

    return (x_train, y_train), (x_test, y_test)

def train_model(epochs, model_dir, train_data_dir):
    (x_train, y_train), (x_test, y_test) = load_cifar10_data(train_data_dir)
    model = create_model()
    model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

    model_dir = "/opt/ml/model"
    versioned_model_dir = os.path.join(model_dir, "1")  # Add numeric subdirectory
    os.makedirs(versioned_model_dir, exist_ok=True)
    model.save(versioned_model_dir, save_format="tf")  # Save in SavedModel format

if __name__ == "__main__":
    train_data_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train the model")
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model", help="Path to save the trained model")
    args = parser.parse_args()

    print(f"Training data directory: {train_data_dir}")
    print(f"Model directory: {args.model_dir}")

    # Train the model
    train_model(args.epochs, args.model_dir, train_data_dir)
