import os
import cv2
import numpy as np
import random
import json

IMG_SIZE = 512
random.seed(42)
np.random.seed(42)

BACKGROUND_PATH = 'background.jpg'
background_img = cv2.imread(BACKGROUND_PATH)
if background_img is None:
    raise FileNotFoundError(f"找不到背景圖檔：{BACKGROUND_PATH}")
background_img = cv2.resize(background_img, (IMG_SIZE, IMG_SIZE))

def generate_image_and_labels(save_dir, img_id):
    img = background_img.copy()
    labels = []

    shapes = ['rectangle', 'circle']
    num_shapes = random.randint(2, 4)

    for _ in range(num_shapes):
        shape = random.choice(shapes)

        if shape == 'rectangle':
            color = (0, 165, 255)  # 橘色 (BGR: 橘)
            w = random.randint(40, 80)
            h = random.randint(40, 80)
            margin = 40
            cx = random.randint(margin, IMG_SIZE - margin)
            cy = random.randint(margin, IMG_SIZE - margin)

            x1 = int(np.clip(cx - w / 2, 0, IMG_SIZE - 1))
            y1 = int(np.clip(cy - h / 2, 0, IMG_SIZE - 1))
            x2 = int(np.clip(cx + w / 2, 0, IMG_SIZE - 1))
            y2 = int(np.clip(cy + h / 2, 0, IMG_SIZE - 1))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            labels.append({"class": 1, "bbox": [x1, y1, x2, y2]})

        elif shape == 'circle':
            color = (255, 0, 0)  # 藍色 (BGR: 藍)
            radius = random.randint(20, 40)
            margin = radius + 20
            cx = random.randint(margin, IMG_SIZE - margin)
            cy = random.randint(margin, IMG_SIZE - margin)

            x1 = cx - radius
            y1 = cy - radius
            x2 = cx + radius
            y2 = cy + radius

            cv2.circle(img, (cx, cy), radius, color, -1)
            labels.append({"class": 2, "bbox": [x1, y1, x2, y2]})

    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

    img_path = os.path.join(save_dir, "images", f"{img_id:04d}.png")
    label_path = os.path.join(save_dir, "labels", f"{img_id:04d}.json")

    cv2.imwrite(img_path, img)

    for label in labels:
        label["class"] = int(label["class"])
        label["bbox"] = [int(x) for x in label["bbox"]]

    with open(label_path, 'w') as f:
        json.dump(labels, f)

def generate_dataset():
    generate_count = {
        "train": 100,
        "val": 25
    }

    for split, count in generate_count.items():
        save_dir = f"./dataset/{split}"
        for img_id in range(count):
            generate_image_and_labels(save_dir, img_id)

if __name__ == "__main__":
    generate_dataset()
