# Common imports
import os
import numpy as np
import pandas as pd
import keras

os.environ["KERAS_BACKEND"] = "tensorflow"

import matplotlib.pyplot as plt

# Load the data
cifar10 = keras.datasets.cifar10
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()

print("Full training set shape:", X_train_full.shape)
print("Test set shape:", X_test.shape)

y_train_full = y_train_full.flatten()
y_test = y_test.flatten()

# Split into validation + training set, and scale pixel intensities to 0-1
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

model = keras.models.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Conv2D(filters=64, kernel_size=7, activation='relu', padding="SAME"),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding="SAME"),
    keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding="SAME"),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding="SAME"),
    keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding="SAME"),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=10, activation='softmax'),
])

model.summary()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(momentum=0.9),
              metrics=["accuracy"])

# Train
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

# learning curves
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.title("Baseline model - learning curves")
plt.savefig("v1_baseline_learning_curves.png")
plt.show()

# Evaluate on test set

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nBASELINE MODEL RESULTS")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")