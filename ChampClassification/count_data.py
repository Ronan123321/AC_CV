import os

base_folder = "FeatureMatch/champs"

folder_counts = []
for folder in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder)
    if os.path.isdir(folder_path):
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        count = len(files)
        folder_counts.append((folder, count))

folder_counts.sort(key=lambda x: (x[1] != 0, x[1]))

for folder, count in folder_counts:

    if count == 0:
        print(f"{folder}: empty")
    else:
        if count < 50:
            print(f"{folder}: {count} files")