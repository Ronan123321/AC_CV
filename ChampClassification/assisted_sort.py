import os
import torch
from torchvision import transforms, models
from PIL import Image, ImageTk
import tkinter as tk

source_folder = "ChampImages"
destination_folder = "FeatureMatch/champs"
num_classes = 71  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("finetuned_resnet50.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_champs = sorted(os.listdir(destination_folder))
image_files = [f for f in os.listdir(source_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
index = 0

def predict_class(img_path):
    pil_img = Image.open(img_path).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    return valid_champs[predicted.item()]

def load_image():
    global img_display, image_label
    img_path = os.path.join(source_folder, image_files[index])
    img = Image.open(img_path)
    img.thumbnail((400, 400))
    img_display = ImageTk.PhotoImage(img)
    image_label.config(image=img_display)

    predicted = predict_class(img_path)
    champ_entry.delete(0, tk.END)
    champ_entry.insert(0, predicted)

def save_label():
    champ_name = champ_entry.get().strip().lower()
    champ_folder = os.path.join(destination_folder, champ_name)
    if not os.path.exists(champ_folder):
        print(f"Folder '{champ_name}' does not exist.")
        return
    src_path = os.path.join(source_folder, image_files[index])
    dst_path = os.path.join(champ_folder, image_files[index])
    os.rename(src_path, dst_path)
    next_image()

def next_image():
    global index
    index += 1
    if index < len(image_files):
        champ_entry.delete(0, tk.END)
        load_image()
    else:
        image_label.config(text="Finished")

def update_suggestions(event):
    typed = champ_entry.get().strip().lower()
    listbox.delete(0, tk.END)
    if typed:
        for champ in valid_champs:
            if champ.startswith(typed):
                listbox.insert(tk.END, champ)
    else:
        for champ in valid_champs:
            listbox.insert(tk.END, champ)

def autofill(event):
    if listbox.size() > 0:
        champ_entry.delete(0, tk.END)
        champ_entry.insert(0, listbox.get(0))
        listbox.delete(0, tk.END)
    return "break"

def on_enter(event):
    save_label()
    return "break"

root = tk.Tk()
root.title("Assisted Champion Image Labeler")

tk.Button(root, text="Accept/Move", command=save_label).pack()
image_label = tk.Label(root)
image_label.pack()

champ_entry = tk.Entry(root, width=30)
champ_entry.pack()

listbox = tk.Listbox(root, height=5)
listbox.pack()

champ_entry.bind("<KeyRelease>", update_suggestions)
champ_entry.bind("<Tab>", autofill)
champ_entry.bind("<Return>", on_enter)

load_image()
root.mainloop()