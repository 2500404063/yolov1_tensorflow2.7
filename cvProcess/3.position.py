import json
import os
import pickle
import cv2 as cv
import numpy as np

"""
Author: Felix
Subscription:
    1. n: next picture
    2. Left Button: Point at the left top angle
    3. Right Button: Point at the right bottom angle and print data for YOLO
    4. p: print current rect
    5. o: output to file(pickle format)
    6. esc: exit
"""


# Definitions:
class Config:
    Selection_rect_color = (128, 0, 200)
    Selection_rect_thickness = 2
    Borders_amount = (8, 6)
    Picture_scale_rate = (4, 4)
    Input_dir = './res_processed_0'
    Output_dir = os.path.join(Input_dir, 'data')


# Global variables
hasStarted = False
point_start = (0, 0)
point_end = (0, 0)
file_name = None
img = None
w = None
h = None
data = {
    'img': None,
    'class': [],
    'x': [],
    'y': [],
    'w': [],
    'h': []
}
label_index = 0


def onMouse(event, x, y, flag, parameters):
    global hasStarted
    global point_start
    global point_end
    if event == cv.EVENT_LBUTTONUP:
        hasStarted = True
        point_start = (x, y)
    elif event == cv.EVENT_MOUSEMOVE and hasStarted == True:
        img_copy = np.array(img)
        cv.rectangle(img_copy, point_start, (x, y),
                     Config.Selection_rect_color, Config.Selection_rect_thickness)
        cv.imshow('main', img_copy)
    elif event == cv.EVENT_RBUTTONUP:
        point_end = (x, y)
        img_copy = np.array(img)
        cv.rectangle(img_copy, point_start, point_end,
                     Config.Selection_rect_color, Config.Selection_rect_thickness)
        cv.imshow('main', img_copy)
        hasStarted = False


def FigureAndGenerate():
    point_mid = (int((point_start[0]+point_end[0])/2),
                 int((point_start[1]+point_end[1])/2))
    # For traditional YOLO dataset: Value: [0,1]
    # rate_mid = (point_mid[0] / w, point_mid[1] / h)
    # rate_wh = (abs(end[0]-start[0]) / w, abs(end[1]-start[1]) / h)
    # print(rate_mid[0], rate_mid[1], rate_wh[0], rate_wh[1])

    # For absolute position:
    point_mid = (int(point_mid[0] / Config.Picture_scale_rate[0]),
                 int(point_mid[1] / Config.Picture_scale_rate[1]))
    wh = (int(abs(point_end[0]-point_start[0]) / Config.Picture_scale_rate[0]),
          int(abs(point_end[1]-point_start[1]) / Config.Picture_scale_rate[1]))
    print(file_name, label_index, point_mid[0], point_mid[1], wh[0], wh[1])
    data['img'] = file_name
    data['class'].append(label_index)
    data['x'].append(point_mid[0])
    data['y'].append(point_mid[1])
    data['w'].append(wh[0])
    data['h'].append(wh[1])


if __name__ == '__main__':
    files = os.listdir(Config.Input_dir)
    parts = Config.Borders_amount  # (Width, Height)
    scale = Config.Picture_scale_rate  # (Fact_X, Fact_Y)
    if not os.path.exists(Config.Output_dir):
        os.mkdir(Config.Output_dir)
    cv.namedWindow('main')
    for file_name in files:
        if file_name == 'data':
            continue
        if os.path.exists(os.path.join(Config.Output_dir, file_name + ".json")):
            continue
        print(f"Opened {file_name}")
        img = cv.imread(os.path.join(Config.Input_dir, file_name))
        img = cv.resize(img, None, None, scale[0], scale[1], cv.INTER_LINEAR)
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
            cv.line(img, startPoint, endPoint, (255, 128, 128), 1)
        cv.imshow('main', img)
        cv.setMouseCallback('main', onMouse, None)
        while True:
            key = cv.waitKey()
            if key == ord('n'):
                data = {
                    'img': None,
                    'class': [],
                    'x': [],
                    'y': [],
                    'w': [],
                    'h': []
                }
                label_index = 0
                break
            elif key == ord('p'):
                cv.rectangle(img, point_start, point_end,
                             Config.Selection_rect_color, Config.Selection_rect_thickness)
                FigureAndGenerate()
                label_index = label_index + 1
            elif key == ord('o'):
                with open(os.path.join(Config.Output_dir, file_name + ".json"), "w+") as f:
                    json.dump(data, f)
                print(f'Wrote {file_name + ".json"} successfully')
            elif key == 27:
                exit(0)
            # else:
            #     print("No this key", key)
