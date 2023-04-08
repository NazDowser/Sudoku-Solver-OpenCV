import cv2
import numpy as np
from matplotlib import pyplot as plt

import preprocess
import matrixSeg
import knn
import sudokuRecog
import sudokuSolver

def devRun():
    for i in range(1,3):
        print(f"############# Question {i} #############")
        flag, sudoku = sudokuRecog.sudokuRecog(i)
        if flag == False:
            print('FALSE READING')
            continue
        for i in range(9):
            print(sudoku[i])
        if sudokuSolver.sudoku_solve(sudoku, 0, 0):
            print("############# solving #############")
            for i in range(9):
                print(sudoku[i])
        else:
            print("NO answer")
            for i in range(9):
                print(sudoku[i])