#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install ultralytics')
get_ipython().system('pip install opencv-python-headless')


# In[ ]:


import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


# In[ ]:


model = YOLO('yolov8x.pt')


# In[ ]:


def detect_objects(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV BGR formatını RGB'ye çevir
    results = model(image)[0]

    detections = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        detections.append([int(x1), int(y1), int(x2), int(y2), round(score, 3),
                           results.names[int(class_id)]])

    return detections, image


# In[ ]:


def plot_detections(image, detections):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    for detection in detections:
        x1, y1, x2, y2, score, class_name = detection
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        plt.text(x1, y1, f'{class_name} {score}', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=12, color='black')

    plt.axis('off')
    plt.show()


# In[ ]:


import os
import glob

# Görüntü dosyalarının bulunduğu dizin
image_directory = '/kaggle/input/new-animals/Photo Animal'

# Dizindeki tüm görüntü dosyalarının yollarını al
image_paths = glob.glob(os.path.join(image_directory, '*'))

# Tespit edilen nesneleri ve görüntüleri işlemek için döngü
for image_path in image_paths:
    detections, image = detect_objects(image_path)
    print(detections)


# In[ ]:


import os
import glob

# Görüntü dosyalarının bulunduğu dizin
image_directory = '/kaggle/input/new-animals/Photo Animal'

# Dizindeki tüm görüntü dosyalarının yollarını al
image_paths = glob.glob(os.path.join(image_directory, '*'))

# Tespit edilen nesneleri ve görüntüleri işlemek için döngü
for image_path in image_paths:
    detections, image = detect_objects(image_path)
    plot_detections(image, detections)
    print(detections)

