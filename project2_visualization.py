# Step 5: Model Testing on Specific Images
IMAGE_SIZE = (500, 500)
TEST_IMAGES = [
    ("test/crack/test_crack.jpg", "crack"),
    ("test/missing-head/test_missinghead.jpg", "missing-head"),
    ("test/paint-off/test_paintoff.jpg", "paint-off"),
]


from tensorflow import keras

mdl_1 = keras.models.load_model("model1.keras")
mdl_2 = keras.models.load_model("model2.keras")

from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE, color_mode="grayscale")
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)
temp_gen = datagen.flow_from_directory(
    "test", target_size=IMAGE_SIZE, color_mode="grayscale",
    class_mode="categorical", shuffle=False, batch_size=1
)
idx_to_name = {v: k for k, v in temp_gen.class_indices.items()}
del temp_gen


def show_predictions(model, model_name):
    plt.figure(figsize=(12, 6))
    for i, (img_path, true_label) in enumerate(TEST_IMAGES):

        x, img_display = preprocess_img(img_path)
        preds = model.predict(x, verbose=0)[0]
        pred_label = idx_to_name[np.argmax(preds)]
 
        
        plt.subplot(1, 3, i + 1)
        plt.imshow(img_display, cmap="gray")
        plt.axis("off")
        color = "green" if pred_label == true_label else "red"
        plt.title(f"P:{pred_label}\nT:{true_label}", color=color, fontsize=20)

        y_text = 480  
        for j, prob in enumerate(preds):
            class_name = idx_to_name[j]
            conf = prob * 100
            plt.text(
                5, y_text, f"{class_name}: {conf:.1f}%", 
                color="lime" if j == np.argmax(preds) else "white",
                fontsize=20, ha="left", va="bottom",
                bbox=dict(facecolor="black", alpha=0.5, pad=1)
            )
            y_text -= 60

    plt.suptitle(f"{model_name} Predictions on Specific Test Images", fontsize=20)
    plt.tight_layout()
    plt.show()


show_predictions(mdl_1, "Model 1 (Shallow CNN)")
show_predictions(mdl_2, "Model 2 (Deeper CNN)")
