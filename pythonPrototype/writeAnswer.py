import cv2
import numpy as np
from matplotlib import pyplot as plt

SIDE = 810
CELLSIDE = int(SIDE / 9)

def writeAnswer(img, answers, validList):
    dim = (SIDE, SIDE)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2.5
    color = (255, 0, 0)
    thickness = 5

    for row in range(9):
        for col in range(9):
            if validList[row*9+col] == 0:
                answer = answers[row][col]
                x = col * CELLSIDE + 15
                y = (row+1) * CELLSIDE - 15
                org = (x, y)
                img = cv2.putText(img, str(answer), org, font, fontScale, color, thickness, cv2.LINE_AA)
    return img