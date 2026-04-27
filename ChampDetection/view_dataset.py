import os
import cv2
from dotenv import load_dotenv

load_dotenv()

image_dir = os.getenv("IMAGE_DIR")
label_dir = os.getenv("LABEL_DIR")
class_names = ["champion"]  

def draw_labels(image_path, label_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    with open(label_path, 'r') as f:
        for line in f.readlines():
            cls_id, x, y, bw, bh = map(float, line.strip().split())
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = class_names[int(cls_id)]
            cv2.putText(img, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Labeled Image", img)
    cv2.waitKey(0)

def main():
    for label_name in os.listdir(label_dir):
        image_name = label_name[:label_name.find("txt")] + "jpg"
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, label_name)
        draw_labels(image_path, label_path)

if __name__ == "__main__":
    main()