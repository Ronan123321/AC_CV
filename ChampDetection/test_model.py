import torch
import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp14/weights/best.pt', force_reload=True)

print(model.names)
print(len(model.names)) 

device = torch.device(0)
model.cuda()  # GPU
model.to(device)

im1 = cv2.imread("test_explicit/Screenshot_orb_7.png")[..., ::-1]
im2 = cv2.imread("test_explicit/Fullscreen_2_4_cost.png")[..., ::-1]  

results = model([ im1, im2 ], size=640)  


results.print()
results.show()  

results.xyxy[0] 
results.pandas().xyxy[0]  # im1 predictions 