import numpy as np
import glob
import cv2
from collections import Counter

TRAINSIZE = 21

def k_nearest_neighbors(image, train_images, train_labels, k):
    '''
    Parameters:
    image - one image
    train_images - a list of N images
    train_labels - a list of N labels corresponding to the N images
    k - the number of neighbors to return

    Output:
    neighbors - 1-D array of k images, the k nearest neighbors of image
    labels - 1-D array of k labels corresponding to the k images
    '''
    imageSetSize = len(train_images)
    
    distances = (np.tile(image, (imageSetSize, 1)) - train_images) ** 2
    distances = np.sum(distances, axis=1)
    sortedIdx = np.argsort(distances)
    # neighbors = train_images[sortedIdx[:k]]
    labels = train_labels[sortedIdx[:k]]

    return labels

def classify_devset(dev_images, train_images, train_labels, k):
    '''
    Parameters:
    dev_images (list) -M images
    train_images (list) -N images
    train_labels (list) -N labels corresponding to the N images
    k (int) - the number of neighbors to use for each dev image

    Output:
    hypotheses (list) -one majority-vote labels for each of the M dev images
    '''
    # majorRatio = k/2
    hypotheses = np.zeros(len(dev_images))
    scores = np.zeros(len(dev_images))

    for devIdx in range(len(dev_images)):
        dev_img = dev_images[devIdx]
        labels = k_nearest_neighbors(dev_img, train_images, train_labels, k)
        # d = {}
        # for l in labels:
        #     d[l] = 
        hypotheses[devIdx] = Counter(labels).most_common(1)[0][0]
        # if hypotheses[devIdx] == 1:
        #     print(labels)

    return hypotheses, scores

def accuracy_estimate(hypotheses, references):
    '''
    Parameters:
    hypotheses (list) - a list of M labels output by the classifier
    references (list) - a list of the M correct labels

    Output:
    accuracy (float) - accuracy of hypotheses
    '''
    diff = np.subtract(hypotheses, references)
    tp = (diff == 0).sum()
    accuracy = tp / len(hypotheses)
    return accuracy

def confusion_matrix(hypotheses, references):
    '''
    Parameters:
    hypotheses (list) - a list of M labels output by the classifier
    references (list) - a list of the M correct labels

    Output:
    confusions (list of lists, or 2d array) - confusions[m][n] is 
    the number of times reference class m was classified as
    hypothesis class n.
    accuracy (float) - the computed accuracy
    f1(float) - the computed f1 score from the matrix
    '''
    confusion = np.zeros((2,2))

    for i in range(len(hypotheses)):
        p = int(hypotheses[i])
        a = int(references[i])
        confusion[a][p] += 1
    tp = confusion[1][1]
    tn = confusion[0][0]
    fp = confusion[0][1]
    fn = confusion[1][0]
    accuracy = (tp+tn) / (tp+tn+fp+fn)
    reRecall = (tp+fn) / tp
    rePrecision = (fp+tp) / tp
    f1 = 2 / (reRecall+rePrecision)

    return confusion, accuracy, f1

def dataSetLoad():
    '''
    Parameters:
    nothing

    Output:
    train_images (list) - a list of N images
    train_labels (list) - a list of N labels corresponding to the N images
    '''
    label = 1
    idx = 0
    train_images = []
    train_labels = []
    
    for file_path in glob.glob("./trainSameFont/[1-9]/*.png"):
        if idx == TRAINSIZE:
            idx = 0
            label += 1
        
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.resize(img, (20,20), interpolation = cv2.INTER_AREA)
        train_images.append(img2.flatten())
        train_labels.append(label)

        idx += 1
    print('############## train set loaded ##############')
    train_labels = np.array(train_labels)
    return train_images, train_labels
        