import os
import shutil
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from dotenv import load_dotenv



source_folder = "ChampImages"
destination_folder = "FeatureMatch/champs"

load_dotenv()
valid_champs = [
    "aatrox", "ahri", "akali", "ashe", "braum", "caitlyn", "clone", "darius", "dr.mundo", "ekko", "ezreal",
    "gangplank", "garen", "gnar", "gwen", "janna", "jarvan_iv", "jayce", "jhin", "jinx", "kai'sa",
    "kalista", "karma", "katarina", "kayle", "kennen", "kobuko", "kog'maw", "k'sante", "lee_sin",
    "leona", "lucian", "lulu", "lux", "malphite", "malzahar", "mighty_mech", "naafiri", "neeko", "poppy", "rakan",
    "rammus", "rell", "ryze", "samira", "senna", "seraphine", "sett", "shen", "sivir", "smolder",
    "swain", "syndra", "twisted_fate", "udyr", "varus", "vi", "viego", "volibear", "xayah",
    "xin_zhao", "yasuo", "yone", "yuumi", "zac", "ziggs", "zyra", "zyra_plant", "dummy", "trash"
]

image_files = [f for f in os.listdir(source_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
index = 0

def load_image():
    global img_display, image_label
    img_path = os.path.join(source_folder, image_files[index])
    img = Image.open(img_path)
    img.thumbnail((400, 400))
    img_display = ImageTk.PhotoImage(img)
    image_label.config(image=img_display)

def save_label():
    champ_name = champ_entry.get().strip().lower()
    champ_folder = os.path.join(destination_folder, champ_name)

    if not os.path.exists(champ_folder):
        print(f"Folder '{champ_name}' does not exist.")
        return

    src_path = os.path.join(source_folder, image_files[index])
    dst_path = os.path.join(champ_folder, image_files[index])
    shutil.move(src_path, dst_path)

    # champ_entry.delete(0, tk.END)

    next_image()

def next_image():
    global index
    index += 1
    if index < len(image_files):
        champ_entry.delete(0, tk.END)
        load_image()
    else:
        image_label.config(text="all images labeled!")

def update_suggestions(event):
    typed = champ_entry.get().strip().lower()
    suggestions = [name for name in valid_champs if name.startswith(typed)]

    listbox.delete(0, tk.END)
    for suggestion in suggestions:
        listbox.insert(tk.END, suggestion)

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
root.title("Champion Image Labeler")

tk.Button(root, text="Label and Move", command=save_label).pack()
image_label = tk.Label(root)
image_label.pack()

champ_entry = tk.Entry(root, width=30)
champ_entry.pack()

listbox = tk.Listbox(root, height=5)
listbox.pack()

# bind keys
champ_entry.bind("<KeyRelease>", update_suggestions)
champ_entry.bind("<Tab>", autofill)

champ_entry.bind("<Return>", on_enter)

load_image()
root.mainloop()