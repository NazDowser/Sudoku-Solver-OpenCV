import preprocess
import matrixSeg
import knn
import cv2
import numpy as np
from matplotlib import pyplot as plt

debug = 0

def sudokuRecog(flag):
    # loading image
    if flag == 1:
        img = cv2.imread('inSameFont/dev1.jpg')
    else:
        img = cv2.imread('inSameFont/dev2.jpg')
    # img = cv2.imread('in/test4.jpg')
    
    # filtering
    imgThresh = preprocess.filtering(img)
    warped, imgMask = preprocess.correction(imgThresh)
    if debug == 1:
        plt.imshow(warped, "gray")
    
    # sementation
    imgSet, validList = matrixSeg.cellSegment(warped)
    dev_images = []
    for i in imgSet:
        for k in i:
                dev_images.append(k.flatten())
                
    # knn
    if flag == 1:
        actural =  [[7, 2, 1, 8, 4, 0, 0, 0, 0],
                    [3, 0, 8, 2, 0, 9, 6, 0, 7],
                    [5, 0, 0, 0, 0, 7, 0, 0, 4],
                    [2, 0, 0, 1, 0, 4, 8, 0, 0],
                    [4, 3, 9, 0, 0, 0, 0, 0, 5],
                    [0, 8, 7, 0, 0, 0, 3, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 5, 8],
                    [0, 5, 2, 0, 7, 0, 0, 3, 0],
                    [9, 0, 3, 5, 0, 0, 0, 0, 0]]
    else:
        actural =  [[0, 0, 6, 0, 0, 2, 0, 3, 0],
                    [0, 8, 0, 0, 5, 0, 0, 6, 2],
                    [2, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 8, 0, 0, 0],
                    [0, 0, 4, 1, 0, 0, 0, 8, 3],
                    [7, 0, 0, 0, 9, 0, 0, 0, 5],
                    [8, 2, 0, 0, 0, 9, 0, 5, 6],
                    [0, 9, 0, 8, 0, 0, 3, 0, 0],
                    [6, 0, 0, 7, 0, 4, 0, 9, 1]]
    train_images, train_labels = knn.dataSetLoad()
    lowest = 50
    good = None
    k_small = 0
    for k in range(1,20): 
        hyp, scores = knn.classify_devset(dev_images, train_images, train_labels, k=k)
        hypotheses = np.zeros((9,9))
        i = 0
        for y in range(9):
            for x in range(9):
                if validList[y*9+x] == 1:
                    hypotheses[y][x] = hyp[i]
                    i += 1
        diff = hypotheses - actural
        for row in diff:
            a = np.count_nonzero(diff)
        if a < lowest:
            lowest = a
            good = hypotheses
            k_small = k

    diff2 = good - actural
    if debug == 1:
        print('############## best hypothesis ##############')
        print('k =', k_small)
        print(good)
        print('############## difference estimate ##############')
        print('difference: ', a)
        if a != 0:
            print(diff2)
    
    

    # hyp, scores = knn.classify_devset(dev_images, train_images, train_labels, k=3)
    # hypotheses = np.zeros((9,9))
    # i = 0
    # for y in range(9):
    #     for x in range(9):
    #         if validList[y*9+x] == 1:
    #             hypotheses[y][x] = hyp[i]
    #             i += 1
    # print(hypotheses)
    
    
    if lowest == 0:
        return True, good
    else:
        return False, None