from tensorflow.keras.preprocessing import image_dataset_from_directory

train_ds = image_dataset_from_directory(
    "data/train",
    image_size=(224, 224),
    batch_size=32
)

val_ds = image_dataset_from_directory(
    "data/val",
    image_size=(224, 224),
    batch_size=32
)

print("ONLY IN TRAIN:")
print(set(train_ds.class_names) - set(val_ds.class_names))

print("ONLY IN VAL:")
print(set(val_ds.class_names) - set(train_ds.class_names))