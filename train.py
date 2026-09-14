import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =====================
# SETTINGS
# =====================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

train_dir = "data/train"
val_dir = "data/val"
test_dir = "data/test"

# =====================
# LOAD DATASET
# =====================
train_ds = image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

# =====================
# PREPROCESS INPUT (IMPORTANT FIX)
# =====================
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))
test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# =====================
# BASE MODEL (MobileNetV2)
# =====================
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # first stage frozen

# =====================
# MODEL BUILD
# =====================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

# =====================
# COMPILE (STAGE 1)
# =====================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =====================
# TRAIN STAGE 1
# =====================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# =====================
# FINE TUNING
# =====================
print("Starting fine-tuning...")

base_model.trainable = True

# optional: freeze early layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# =====================
# EVALUATION
# =====================
test_loss, test_acc = model.evaluate(test_ds)
print("Test Accuracy:", test_acc)

# =====================
# SAVE MODEL (NEW FORMAT)
# =====================
model.save("dog_breed_model.keras")