import os
import cv2 as cv
import numpy as np

input_dir = './res_original'
output_dir = './res_processed_0'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
files = os.listdir(input_dir)
for file in files:
    img = cv.imread(os.path.join(input_dir, file))
    img = cv.resize(img, (256, 192))
    cv.imwrite(os.path.join(output_dir, file), img)
