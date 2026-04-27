import os
import time
import torch
from torchvision import models, transforms
import cv2
import numpy as np
from mss import mss
from PIL import Image, ImageDraw, ImageTk
import win32gui
import win32con
import tkinter as tk
from dotenv import load_dotenv

load_dotenv()

yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='ChampDetectionModel/finetuned_yolov5s_V3.pt')
yolo_model.eval()

num_classes = 71 

# resnet 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = models.resnet50(pretrained=False)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, num_classes)
resnet_model.load_state_dict(torch.load('finetuned_resnet50.pth', map_location=device))
resnet_model = resnet_model.to(device)
resnet_model.eval()

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# screen cap
TARGET_WINDOW = os.getenv("WINDOW_NAME")
hwnd = win32gui.FindWindow(None, TARGET_WINDOW)
if hwnd == 0:
    raise Exception(f"Window '{TARGET_WINDOW}' not found!")

left, top, right, bottom = win32gui.GetWindowRect(hwnd)
width, height = right - left, bottom - top
monitor = {"top": top, "left": left, "width": width, "height": height}
sct = mss()

class ChampArea:
    def __init__(self, x1, y1, x2, y2, label):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.label = label

# run model
def inference_loop():
    sct_img = sct.grab(monitor)
    frame = np.array(sct_img)  
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

    results = yolo_model(frame_rgb)
    detections = results.xyxy[0]  

    champ_areas = []
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        img_pil = Image.fromarray(crop)
        tensor = resnet_transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = resnet_model(tensor)
            pred_class = preds.argmax(dim=1).item()

        champ_areas.append(ChampArea(x1, y1, x2, y2, pred_class))

    return champ_areas

# transparent overlay
root = tk.Tk()
root.overrideredirect(True)
root.geometry(f"{width}x{height}+{left}+{top}")
root.attributes('-topmost', True)
root.attributes('-transparentcolor', '#ff00ff')  

canvas = tk.Canvas(root, width=width, height=height, highlightthickness=0, bd=0, bg='#ff00ff')
canvas.pack()

def apply_click_through():
    hwnd_overlay = root.winfo_id()
    extended_style = win32gui.GetWindowLong(hwnd_overlay, win32con.GWL_EXSTYLE)
    win32gui.SetWindowLong(
        hwnd_overlay,
        win32con.GWL_EXSTYLE,
        extended_style | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
    )

def update_overlay():
    global last_update, champ_list
    now = time.time()
    if now - last_update > 1:
        champ_list = inference_loop()
        last_update = now

        overlay_img = Image.new("RGBA", (width, height), (255, 0, 255, 255))
        draw = ImageDraw.Draw(overlay_img)

        for champ in champ_list:
            draw.rectangle([champ.x1, champ.y1, champ.x2, champ.y2], outline="red", width=3)
            draw.text((champ.x1 + 5, champ.y1 - 15), f"Class: {champ.label}", fill="white")

        tk_img = ImageTk.PhotoImage(overlay_img)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=tk_img)
        canvas.image = tk_img

    root.after(33, update_overlay)

last_update = 0
champ_list = []

def key_event(event):
    if event.keysym == "Escape":
        root.quit()

root.bind("<KeyPress>", key_event)

update_overlay()
root.after(200, apply_click_through)

root.mainloop()