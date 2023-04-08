import cv2
import numpy as np
from matplotlib import pyplot as plt

TOPLEFT = 0
TOPRGHT = 1
BOTLEFT = 2
BOTRGHT = 3

def coordinateUnpack(coordinates):
    '''
    INPUT:
    coordinates (2d list) - unordered simple contour coordinate list
    
    OUTPUT:
    rect (2d list) - ordered coordinate list
    '''
    rect = np.zeros((4, 2), dtype = "float32")
    
    s = np.sum(coordinates, axis = 1)
    rect[TOPLEFT] = coordinates[np.argmin(s)] # top left
    rect[BOTRGHT] = coordinates[np.argmax(s)] # bottom right
    diff = np.diff(coordinates, axis = 1)
    rect[TOPRGHT] = coordinates[np.argmin(diff)] # top right
    rect[BOTLEFT] = coordinates[np.argmax(diff)] # bottom left
    
    return rect

def harmongraphyMatrix(img, coordinates):
    '''
    INPUT:
    img (mat) - arbitrary img
    coordinates (2d list) - unordered simple contour coordinate list
    
    OUTPUT:
    warped (mat) - angle corrected image
    '''
    rect = coordinateUnpack(coordinates)
    
    # calculate max actual width
    widthTop = np.sqrt(((rect[TOPRGHT][0] - rect[TOPLEFT][0]) ** 2) + ((rect[TOPRGHT][1] - rect[TOPLEFT][1]) ** 2))
    widthBot = np.sqrt(((rect[BOTRGHT][0] - rect[BOTLEFT][0]) ** 2) + ((rect[BOTRGHT][1] - rect[BOTLEFT][1]) ** 2))
    maxWidth = max(int(widthTop), int(widthBot))
    
    # calculate max height
    heightLEFT = np.sqrt(((rect[BOTLEFT][0] - rect[TOPLEFT][0]) ** 2) + ((rect[BOTLEFT][1] - rect[TOPLEFT][1]) ** 2))
    heightRGHT = np.sqrt(((rect[BOTRGHT][0] - rect[TOPRGHT][0]) ** 2) + ((rect[BOTRGHT][1] - rect[TOPRGHT][1]) ** 2))
    maxHeight = max(int(heightLEFT), int(heightRGHT))
    
    # initialize destination matrix
    dst = np.array(
        [[0, 0],
        [maxWidth - 1, 0],
        [0, maxHeight - 1],
        [maxWidth - 1, maxHeight - 1]],
        dtype = "float32"
    )
    
    # get harmongraphy matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return warped    

def filtering(img):
    '''
    INPUT:
    img (mat) - original img
    
    OUTPUT:
    imgThresh (mat) - filtered binary image
    '''
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.medianBlur(imgGray, 1)
    imgBlur = cv2.GaussianBlur(imgBlur, (3, 3), 0)
    
    # binary color
    imgThresh = cv2.adaptiveThreshold(
        imgBlur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 301, 7
    )
    # plt.imshow(imgThresh, "gray") ######## debug
    
    return imgThresh

def correction(img):
    '''
    INPUT:
    img (mat) - filtered binary image
    
    OUTPUT:
    warped (mat) - angle corrected image
    imgMask (mat) - masked image, used for debug
    '''
    # find ourtermost contours
    contours, __ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # find the contour we want 
    maxArea = 0
    sudokuContor = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > maxArea:
            maxArea = area
            sudokuContor = cnt
            
    # use mask to clean up the area above contour
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, [sudokuContor], 0, 255, -1)
    cv2.drawContours(mask, [sudokuContor], 0, 0, 2)
    imgMask = cv2.bitwise_and(img, mask)
    # plt.imshow(imgMask, "gray") ######## debug
    
    # use outer contour to recover img
    peri = cv2.arcLength(sudokuContor, True)
    approx = cv2.approxPolyDP(sudokuContor, 0.01 * peri, True)
    coordinates = np.array([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
    warped = harmongraphyMatrix(imgMask, coordinates)
    # plt.imshow(warped, "gray") ######## debug
    
    return warped, imgMask

# subject to merger
def correctionOutput(img, imgOriginal):
    '''
    INPUT:
    img (mat) - filtered binary image
    
    OUTPUT:
    warped (mat) - angle corrected image
    imgMask (mat) - masked image, used for debug
    '''
    # find ourtermost contours
    contours, __ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # find the contour we want 
    maxArea = 0
    sudokuContor = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > maxArea:
            maxArea = area
            sudokuContor = cnt
            
    # use mask to clean up the area above contour
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, [sudokuContor], 0, 255, -1)
    cv2.drawContours(mask, [sudokuContor], 0, 0, 2)
    imgMask = cv2.bitwise_and(img, mask)
    # plt.imshow(imgMask, "gray") ######## debug
    
    # use outer contour to recover img
    peri = cv2.arcLength(sudokuContor, True)
    approx = cv2.approxPolyDP(sudokuContor, 0.01 * peri, True)
    coordinates = np.array([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
    warped = harmongraphyMatrix(imgOriginal, coordinates)
    # plt.imshow(warped, "gray") ######## debug
    
    return warped
