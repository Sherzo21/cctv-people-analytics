import os
import cv2
from gender_model import GenderClassifier

IMG_PATH = "test.jpg"  # put the file in the same folder as test_gender.py

print("Current working directory:", os.getcwd())
print("Trying to read:", os.path.abspath(IMG_PATH))

img = cv2.imread(IMG_PATH)
print("Image loaded:", img is not None)

gender = GenderClassifier("weights/best_resnet50_gender_model.pth")

label, conf, probs = gender.predict(img)
print("probs:", probs)   # e.g. [0.99, 0.01]
print("pred:", label, conf)
