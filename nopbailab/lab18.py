import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# Đọc dataset
train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    ".",
    classes=["class1","class2"],
    target_size=(128,128),
    batch_size=2,
    class_mode="categorical",
    subset="training"
)

val_data = train_gen.flow_from_directory(
    ".",
    classes=["class1","class2"],
    target_size=(128,128),
    batch_size=2,
    class_mode="categorical",
    subset="validation"
)

# Load mô hình MobileNetV2 đã được huấn luyện sẵn
base_model = MobileNetV2(
    input_shape=(128,128,3),
    include_top=False,
    weights="imagenet"
)

# Khóa trọng số
base_model.trainable = False

# Tạo mô hình mới
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Huấn luyện
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# Đánh giá
loss, acc = model.evaluate(val_data)

print("Accuracy =", acc)