import numpy as np
import cv2
# image = cv2.imread("sudoku.png")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# sobelX = np.uint8(np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0)))
# sobelY = np.uint8(np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1)))
# sobelCombined = cv2.bitwise_or(sobelX, sobelY)
# cv2.imshow("img", sobelCombined)
# # ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
# #
# # contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # print(len(contours))
# # cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
# # cv2.imshow("img", image)
# cv2.waitKey(0)

def findnextblank (array, x, y):
    for i in range(x, 9):
        for j in range(0, 9):
            if array[i][j] == 0:
                return i, j
    return -1, -1

def itsavailable (array, x, y, number):
    row_check = all(number != array[x][y] for x in range(0, 9))
    if row_check:
        column_check = all(number != array[x][y] for y in range(0, 9))
        if column_check:
            rth = x//3                   # find the 3*3array
            cth = y//3
            for x in range(3*rth, rth*3+3):
                for y in range(3*cth, cth*3+3):
                    if number == array[x][y]:
                        return False
            return True
        else:
            return False
    else:
        return False


def sudoku_solve(array, x, y):
    [x, y] = findnextblank(array, x, y)
    if x == -1:
        return True    # The sudoku is solved
    for n in range(1, 10):
        if itsavailable(array, x, y, n):
            array[x][y] = n
            if sudoku_solve(array, x, y):
                return True
            else:
                array[x][y] = 0
    return False