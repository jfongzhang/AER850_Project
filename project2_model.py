import numpy as np
import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)
keras.utils.set_random_seed(42)

# Step 1: Data Processing
IMAGE_SIZE = (500, 500)
BATCH_SIZE = 8
EPOCHS = 15

train_dir = "train"
test_dir  = "test"
val_dir   = "valid"

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen  = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="grayscale",
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="grayscale",
    shuffle=False
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="grayscale",
    shuffle=False
)

# %%
# Step 2: Neural Network Architecture Design
from tensorflow.keras import layers
# First CNN Model (mdl_1)
mdl_1 = keras.Sequential([
    keras.Input(shape=(500, 500, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    
    layers.Dense(64, activation="relu"),

    layers.Dense(3, activation="softmax")
])

# %%
# SECOND CNN MODEL (mdl_2) Increase complexity 
mdl_2 = keras.Sequential([
    keras.Input(shape=(500, 500, 1)),
    
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    
    layers.Dense(128, activation="relu"),

    layers.Dense(3, activation="softmax")
])

# %%
# Step 3: Hyperparameter Analysis 
# mdl_1
mdl_1.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop_1 = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
)

history1 = mdl_1.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop_1],
    verbose=1
)

# %%
# mdl_2
mdl_2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop_2 = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
)

history2 = mdl_2.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop_2],
    verbose=1
)

# %%
# Step 4: Model Evaluation
# mdl_1
import matplotlib.pyplot as plt

test_loss_1, test_acc_1 = mdl_1.evaluate(test_generator, verbose=0)
print(f"[Model 1] Test accuracy: {test_acc_1:.4f} | Test loss: {test_loss_1:.4f}")

mdl_1.save("model1.keras")

# Plot Accuracy and Loss (mdl_1)
hist1 = history1.history
train_acc_1 = hist1.get("accuracy", hist1.get("categorical_accuracy"))
val_acc_1   = hist1.get("val_accuracy", hist1.get("val_categorical_accuracy"))

plt.figure(figsize=(6,4))
plt.plot(train_acc_1, label="Train Acc (Model 1)")
plt.plot(val_acc_1,   label="Val Acc (Model 1)")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Model 1 Accuracy vs Epoch")
plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(6,4))
plt.plot(hist1["loss"],     label="Train Loss (Model 1)")
plt.plot(hist1["val_loss"], label="Val Loss (Model 1)")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch of Model 1")
plt.legend(); plt.grid(True); plt.show()

# %%
# mdl_2
test_loss_2, test_acc_2 = mdl_2.evaluate(test_generator, verbose=0)
print(f"[Model 2] Test accuracy: {test_acc_2:.4f} | Test loss: {test_loss_2:.4f}")

mdl_2.save("model2.keras")

# Plot Accuracy and Loss (mdl_2)
hist2 = history2.history
train_acc_2 = hist2.get("accuracy", hist2.get("categorical_accuracy"))
val_acc_2   = hist2.get("val_accuracy", hist2.get("val_categorical_accuracy"))

plt.figure(figsize=(6,4))
plt.plot(train_acc_2, label="Train Acc (Model 2)")
plt.plot(val_acc_2,   label="Val Acc (Model 2)")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Model 2 Accuracy vs Epoch")
plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(6,4))
plt.plot(hist2["loss"],     label="Train Loss (Model 2)")
plt.plot(hist2["val_loss"], label="Val Loss (Model 2)")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch of Model 2")
plt.legend(); plt.grid(True); plt.show()