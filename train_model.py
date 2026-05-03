# ============================================================
# Face Mask Detection using CNN - Training Script
# ============================================================

import os
import numpy as np
import kagglehub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report

# ─────────────────────────────────────────
# 1. Data Collection (Kaggle Dataset)
# ─────────────────────────────────────────
print("📥 Downloading dataset...")
path = kagglehub.dataset_download("omkargurav/face-mask-dataset")
data_dir = os.path.join(path, "data")

print("Dataset Path:", path)
print("Folders:", os.listdir(data_dir))

# Check dataset structure
for root, dirs, files in os.walk(data_dir):
    print("Folder:", root)
    print("Subfolders:", dirs)
    print("Number of images:", len(files))
    print("-" * 40)

# ─────────────────────────────────────────
# 2. Data Augmentation & Preprocessing
# ─────────────────────────────────────────
img_size = (224, 224)   # ✅ Increased from 128 → 224 for more detail

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.7, 1.3],   # ✅ Added: handles dark/bright lighting
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2,
)

train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=32,
    class_mode="binary",
    subset="training",
    color_mode="rgb",
)

val_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=32,
    class_mode="binary",
    subset="validation",
    color_mode="rgb",
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_data = test_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=32,
    class_mode="binary",
    shuffle=False,
    color_mode="rgb",
)

print(f"\n✅ Train samples     : {train_data.samples}")
print(f"✅ Validation samples : {val_data.samples}")
print(f"✅ Test samples       : {test_data.samples}")
print(f"✅ Class indices      : {train_data.class_indices}")

# ─────────────────────────────────────────
# 3. Build CNN Model
# ─────────────────────────────────────────
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),  # ✅ 224×224
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 3 — ✅ Extra layer for deeper feature extraction
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Classifier
    Flatten(),
    Dense(256, activation="relu"),   # ✅ Increased from 128 → 256
    Dropout(0.5),
    Dense(1, activation="sigmoid"),  # Binary output
])

model.summary()

# ─────────────────────────────────────────
# 4. Compile
# ─────────────────────────────────────────
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

# ─────────────────────────────────────────
# 5. Callbacks
# ─────────────────────────────────────────
callbacks = [
    # ✅ Stop training when val_loss stops improving for 5 epochs
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    # ✅ Reduce learning rate when stuck
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
    # ✅ Always save the best model
    ModelCheckpoint(
        "face_mask_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]

# ─────────────────────────────────────────
# 6. Train
# ─────────────────────────────────────────
print("\n🚀 Training model (up to 30 epochs with early stopping)...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,           # ✅ Increased from 10 → 30
    callbacks=callbacks,
)

# ─────────────────────────────────────────
# 7. Evaluate
# ─────────────────────────────────────────
print("\n📊 Evaluating on test set...")
loss, accuracy = model.evaluate(test_data)
print(f"\n📊 Test Accuracy : {accuracy:.4f}")
print(f"📊 Test Loss     : {loss:.4f}")

y_pred = (model.predict(test_data) > 0.5).astype(int)
y_true = test_data.classes

print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
print(f"Precision : {precision_score(y_true, y_pred):.4f}")
print(f"Recall    : {recall_score(y_true, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_true, y_pred))

# ─────────────────────────────────────────
# 8. Save Final Model
# ─────────────────────────────────────────
model.save("face_mask_model.h5")
print("\n✅ Model saved as face_mask_model.h5")