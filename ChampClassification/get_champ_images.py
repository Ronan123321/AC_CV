import torch
import cv2
import os

path_to_images = "GameplayImages/"

destination = "ChampImages/"

model = torch.hub.load('ultralytics/yolov5', 'custom', path='ChampDetectionModel/finetuned_yolov5s_V3.pt', force_reload=True)

device = torch.device(0)
model.cuda()  # GPU
model.to(device)

def get_all_champs(img_path):
    img = cv2.imread(img_path)[..., ::-1]

    result = model(img, size=640)
    image_list = []
    
    for champ in result.xyxy[0]:
        if champ[4] > 0.55:
            image_list.append(img[int(champ[1].item()):int(champ[3].item()), 
                                  int(champ[0].item()):int(champ[2].item())].copy())

    return image_list

def get_all_filenames():
    dir_list = os.listdir(path_to_images)
    return dir_list

def main():
    image_names = get_all_filenames()

    for name in image_names:
        champ_images = get_all_champs(path_to_images + name)
        for image in range(len(champ_images)):
            rgb_image = cv2.cvtColor(champ_images[image], cv2.COLOR_RGB2BGR)
            cv2.imwrite(destination + name + "_champ_" + str(image) + ".png", rgb_image)

if __name__ == "__main__":
    main()