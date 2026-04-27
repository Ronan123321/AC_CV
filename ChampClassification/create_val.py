import os
import shutil
import random

source_base = "FeatureMatch/champs"
val_base = "datasets/prelim/val"

os.makedirs(val_base, exist_ok=True)

for champ in os.listdir(source_base):
    champ_src = os.path.join(source_base, champ)
    champ_dst = os.path.join(val_base, champ)
    if os.path.isdir(champ_src):
        os.makedirs(champ_dst, exist_ok=True)

        for old_file in os.listdir(champ_dst):
            old_file_path = os.path.join(champ_dst, old_file)
            if os.path.isfile(old_file_path):
                os.remove(old_file_path)
        files = [f for f in os.listdir(champ_src) if os.path.isfile(os.path.join(champ_src, f))]
        
        val_files = random.sample(files, min(2, len(files)))
        for f in val_files:
            shutil.copy2(os.path.join(champ_src, f), os.path.join(champ_dst, f))