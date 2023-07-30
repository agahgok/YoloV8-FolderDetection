import os
from ultralytics import YOLO
from PIL import Image

model = YOLO('/path/model.pt')

image_folder = "/path/folder"

for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    image = Image.open(image_path)
    
    results = model.predict(source=image, save=True,)
 
