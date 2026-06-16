import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
import numpy as np

# ==========================
# Load CIFAR-10 Dataset
# ==========================
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encoding
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)

# ==========================
# Build CNN Model
# ==========================
model = models.Sequential([
    layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        padding='same',
        input_shape=(32, 32, 3)
    ),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        64,
        (3, 3),
        activation='relu',
        padding='same'
    ),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        64,
        (3, 3),
        activation='relu',
        padding='same'
    ),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),

    layers.Dense(num_classes, activation='softmax')
])

model.summary()

# ==========================
# Compile Model
# ==========================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================
# Train Model
# ==========================
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

# ==========================
# Evaluate Model
# ==========================
test_loss, test_acc = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\nTest Accuracy =", test_acc)

# ==========================
# Prediction Example
# ==========================
predictions = model.predict(x_test)

print(
    "Predicted class:",
    np.argmax(predictions[0])
)

print(
    "Actual class:",
    np.argmax(y_test[0])
)

# ==========================
# Accuracy Graph
# ==========================
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(["Train", "Validation"])

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.legend(["Train", "Validation"])

plt.tight_layout()
plt.show()