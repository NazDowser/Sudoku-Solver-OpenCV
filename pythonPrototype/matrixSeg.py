import cv2
import numpy as np
from matplotlib import pyplot as plt

SIDE = 810
CELLSIDE = int(SIDE / 9)
# CROPSIDE = int(CELLSIDE / 2) 
CROPSIDE = 20
BOUNDTHRES = 4225
NONZEROFACTOR = 150

def trainSetgenerate(warped, initial=0):
    '''
    used for generating train set
    '''
    dim = (SIDE, SIDE)
    warpResized = cv2.resize(warped, dim, interpolation = cv2.INTER_AREA)

    i = initial
    for y in range(9):
        for x in range(9):
            imgCell = warpResized[y*CELLSIDE:(y+1)*CELLSIDE, x*CELLSIDE:(x+1)*CELLSIDE]
            
            # if no more than certain amount of pixels near center, count as 0
            imgCellCrop = imgCell[20:CELLSIDE-30, 25:CELLSIDE-25]
            nonZero = cv2.countNonZero(imgCellCrop)
            if nonZero > NONZEROFACTOR:
                flagBadRead, flagSpace, digit = findSmallestBox(imgCell, True)
                if flagSpace == False:
                    continue
                if flagBadRead == False:
                    print("!!!!!!!!!! errors !!!!!!!!!!")
                    print(y,x)
                    digit = imgCell
                cv2.imwrite(f"./trainSameFont/{i:04d}.png", digit)
                i += 1

def cellSegment(warped):
    '''
    INPUT:
    warped (mat) - angle corrected image
    
    OUTPUT:
    imgSet (mat) - imgSet
    '''
    # uniform size for knn
    dim = (SIDE, SIDE)
    warpResized = cv2.resize(warped, dim, interpolation = cv2.INTER_AREA)

    imgSet = []
    validList = []
    for y in range(9):
        imgRow = []
        for x in range(9):
            imgCell = warpResized[y*CELLSIDE:(y+1)*CELLSIDE, x*CELLSIDE:(x+1)*CELLSIDE]
            
            # if no more than certain amount of pixels near center, count as 0
            imgCellCrop = imgCell[20:CELLSIDE-20, 20:CELLSIDE-20]
            nonZero = cv2.countNonZero(imgCellCrop)
            if nonZero > NONZEROFACTOR:
                flagBadRead, flagSpace, digit = findSmallestBox(imgCell, True)
                if flagSpace == True:
                    # if flagBadRead == False:
                    #     digit = cv2.resize(imgCellCrop, (CROPSIDE,CROPSIDE), interpolation = cv2.INTER_AREA)
                    imgRow.append(digit)
                    validList.append(1)
                else:
                    validList.append(0)
            else:
                validList.append(0)
        imgSet.append(imgRow)

    return imgSet, validList

def findSmallestBox(img, flag):
    '''
    INPUT:
    img (2d list) - single img for digit
    
    OUTPUT:
    digit (mat) - cropped single img for digit
    '''
    maxArea = 0
    sudokuContor = None
    
    # cropping indicator 
    if flag == True:
        img = img[10:CELLSIDE-10, 10:CELLSIDE-10]
    contours, __= cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # using contours to find bound box
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        area = w * h
        
        # neglect grid
        if area > maxArea and area < 4225:
            maxArea = area
            sudokuContor = cnt
            
    # stretch to square shape
    (x, y, w, h) = cv2.boundingRect(sudokuContor)
    if h == 0 or w == 0 or h < 40 :
        return False, True, None
    output = img[y:y+h, x:x+w]
    nonZero = cv2.countNonZero(output)
    
    # # false pos
    # if nonZero < NONZEROFACTOR:
    #     return True, False, None
        
    if h > w:
        ratio = w/h
        newW = int(ratio*CROPSIDE)
        dim = (newW, CROPSIDE)
        outputResized = cv2.resize(output, dim, interpolation = cv2.INTER_AREA)

        digit = np.zeros((CROPSIDE, CROPSIDE), dtype = np.uint8)
        newX = int((CROPSIDE - newW) / 2)
        digit[0:CROPSIDE, newX:newX+newW] = outputResized
    else:
        ratio = h/w
        newH = int(ratio*CROPSIDE)
        dim = (CROPSIDE, newH)
        outputResized = cv2.resize(output, dim, interpolation = cv2.INTER_AREA)
        
        digit = np.zeros((CROPSIDE, CROPSIDE), dtype = np.uint8)
        newY = int((CROPSIDE - newH) / 2)
        digit[newY:newY+newH, 0:CROPSIDE] = outputResized
    
    return True, True, digit