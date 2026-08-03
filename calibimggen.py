import os
import shutil

SOURCE_DIR = r"C:\SignLanguage_Project\dataset_split\train"
DEST_DIR = r"C:\SignLanguage_Project\calib_images"

os.makedirs(DEST_DIR, exist_ok=True)

for cls in sorted(os.listdir(SOURCE_DIR)):

    class_dir = os.path.join(SOURCE_DIR, cls)

    if not os.path.isdir(class_dir):
        continue

    images = sorted([
        f for f in os.listdir(class_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    for img in images[:5]:

        src = os.path.join(class_dir, img)

        dst = os.path.join(
            DEST_DIR,
            f"{cls}_{img}"
        )

        shutil.copy2(src, dst)

print("Calibration images copied.")