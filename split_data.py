import os
import random
import shutil

src_dir= "images/Images"
tgt_dir="data"

train_ratio=0.8
test_ratio=0.1
val_ratio=0.1

random.seed(42)
for breed in os.listdir(src_dir):
    breed_path=os.path.join(src_dir,breed)

    if not os.path.isdir(breed_path):
        continue

    images=os.listdir(breed_path)
    random.shuffle(images)

    total=len(images)

    train_end=int(total* train_ratio)
    val_end= train_end + int(total * val_ratio)

    splits={
        "train": images[:train_end],
        "val" : images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, split_images in splits.items():
        split_breed_dir = os.path.join(tgt_dir, split_name, breed)
        os.makedirs(split_breed_dir, exist_ok=True)

        for img in split_images:
            src = os.path.join(breed_path, img)
            dst = os.path.join(split_breed_dir, img)
            shutil.copy2(src, dst)
print("Done")

