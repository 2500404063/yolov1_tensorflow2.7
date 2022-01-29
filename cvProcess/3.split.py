import os
import cv2 as cv
import numpy as np

input_dir = './res_processed_0'
output_dir = './res_processed_1'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
files = os.listdir(input_dir)
parts = (8, 6)  # (Width, Height)
for file in files:
    if file == 'data':
        continue
    img = cv.imread(os.path.join(input_dir, file))
    w = img.shape[1]
    h = img.shape[0]
    wPartLength = int(w/parts[0])
    hPartLength = int(h/parts[1])
    for i in range(parts[0] - 1):
        startPoint = (wPartLength - 1 + (wPartLength+1) * i, 0)
        endPoint = (wPartLength - 1 + (wPartLength+1) * i, h-1)
        img = cv.line(img, startPoint, endPoint, (255, 128, 128), 1)
    for i in range(parts[1] - 1):
        startPoint = (0, hPartLength - 1 + (hPartLength+1) * i)
        endPoint = (w-1, hPartLength - 1 + (hPartLength+1) * i)
        img = cv.line(img, startPoint, endPoint, (255, 128, 128), 1)
    cv.imwrite(os.path.join(output_dir, file), img)
