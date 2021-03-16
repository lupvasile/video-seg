import os

import cv2

from config.config import Config

cfg = Config.get_config()

dataWin = cv2.namedWindow('data')
labelWin = cv2.namedWindow('label')
cv2.moveWindow('data', 2000, 70)
cv2.moveWindow('label', 2000, 535)
exit = False

for root, dirs, files in os.walk(Config.get_datadir('v_kitti')):
    if 'clone' in dirs:
        dirs[:] = ['clone']
    if 'Camera_0' in dirs:
        dirs[:] = ['Camera_0']
    if 'rgb' in dirs:
        dirs[:] = ['rgb']
    if 'rgb' not in root:
        continue

    files.sort()
    print(root)
    for file in files:
        img = cv2.imread(os.path.join(root, file))
        cv2.imshow('data', img)

        label = cv2.imread(os.path.join(root, file).replace('rgb_', 'classgt_').replace('rgb', 'classSegmentation').replace('jpg', 'png'))
        cv2.imshow('label', label)
        key = cv2.waitKey()
        if key == 119:  # 'w'
            break
        if key == 113:  # 'q'
            exit = True
            break

    if exit == True:
        break
