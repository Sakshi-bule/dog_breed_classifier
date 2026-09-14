import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

IMG_SIZE = (224, 224)

# load model
model = tf.keras.models.load_model("dog_breed_model.keras")

# IMPORTANT:
# use the same class names as training
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/train",
    image_size=IMG_SIZE,
    batch_size=32
)

class_names = train_ds.class_names

# image path (change this)
img_path = "basset2.jpg"

img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

predictions = model.predict(img_array)[0]

top_indices = np.argsort(predictions)[-3:][::-1]

print("Top predictions:")
for i in top_indices:
    print(f"{class_names[i]} : {predictions[i] * 100:.2f}%")