import torch
import os
import shutil
from dotenv import load_dotenv

load_dotenv()

paths_to_data = [ os.getenv("PATHS_TO_DATA")]

image_destination = os.getenv("IMAGE_DEST")
label_destination = os.getenv("LABEL_DEST")

image_file_type = "png"

def moveAllFiles():

    for paths in paths_to_data:
        fileCount = 0
        path_to_img = paths + "images/"
        path_to_label = paths + "labels/"
        for label_name in os.listdir(path_to_label):
            changedClass = ""
            with open(os.path.join(path_to_label, label_name)) as f:
                for line in f:
                    splitText = line.strip().split()
                    splitText[0] = "0"
                    for text in splitText:
                        changedClass += text + " "
                    changedClass = changedClass[:-1] + "\n"

            image_name = label_name[:label_name.find("txt")] + image_file_type
            image_path = os.path.join(path_to_img, image_name)

            if os.path.exists(image_path):
                shutil.copy(image_path, os.path.join(image_destination, image_name))
                f = open(os.path.join(label_destination, label_name), "x")
                f.write(changedClass)
            else:
                print(f"No matching image file found, text '{image_name}' file will not be copied")

            fileCount += 1
            if fileCount >= 1162:
                break

            


def main():
    moveAllFiles()

if __name__ == "__main__":
    main()