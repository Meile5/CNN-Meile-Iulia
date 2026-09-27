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

# V3 changes vs V2:
# - Slightly lower dropout (0.4 instead of 0.5) - BatchNorm already regularizes some,
#   so less aggressive dropout may let the model learn more before being held back
# - Explicit learning rate (0.001, Adam's default, stated explicitly so it's easy to tune)
# - ReduceLROnPlateau callback - automatically lowers the learning rate when validation
#   loss plateaus, often squeezes out extra accuracy in later epochs
# - Larger batch size (64 instead of default 32) - trains faster and can stabilize gradients

DROPOUT_RATE = 0.4
LEARNING_RATE = 0.001
BATCH_SIZE = 64

model = keras.models.Sequential([
    keras.Input(shape=(32, 32, 3)),

    keras.layers.Conv2D(filters=64, kernel_size=7, padding="SAME", use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Conv2D(filters=128, kernel_size=3, padding="SAME", use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(filters=128, kernel_size=3, padding="SAME", use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Conv2D(filters=256, kernel_size=3, padding="SAME", use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(filters=256, kernel_size=3, padding="SAME", use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(units=128, activation="relu"),
    keras.layers.Dropout(DROPOUT_RATE),
    keras.layers.Dense(units=64, activation="relu"),
    keras.layers.Dropout(DROPOUT_RATE),
    keras.layers.Dense(units=10, activation="softmax"),
])

model.summary()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              metrics=["accuracy"])

early_stopping = keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

# Train (more epochs allowed since EarlyStopping will cut it short if val_loss stops improving)
history = model.fit(X_train, y_train, epochs=30, batch_size=BATCH_SIZE,
                     validation_data=(X_valid, y_valid),
                     callbacks=[early_stopping, reduce_lr])

# learning curves
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.title("V3 model - learning curves")
plt.savefig("v3_learning_curves.png")
plt.show()

# Evaluate on test set

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nV3 MODEL RESULTS")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")