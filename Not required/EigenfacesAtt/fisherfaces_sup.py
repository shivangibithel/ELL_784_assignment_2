# Assignment 2 || Fisher Linear Discriminant in Supervised Learning
# Abhishek Ranjan 2019EEA2238
# Prashant Panddy 2019EEA2786

import cv2
import os
import numpy as np


# =========================================================================================
# store images as vector for each class
# =========================================================================================

class Subject:
    def __init__(self, data):
        self.data_subject = data

    def appendData(self, data):
        self.data_subject = np.column_stack((data, self.data_subject))


# =========================================================================================
# finding W_pca
# =========================================================================================

def pca(X, n_components):
    mU_overall = np.mat(X.mean(axis=1)).T
    n_components = check_num_components(n_components)

    X = X - mU_overall
    # to find eigen vectors of XX' we first calculate eigen vectors of X'X
    # then pre-multiply it with matrix X
    XTX = np.dot(X.T, X)

    [e_val_PCA, e_vec_PCA] = np.linalg.eig(XTX)
    e_vec_PCA = np.dot(X, e_vec_PCA)

    for i in range(e_vec_PCA.shape[1]):
        e_vec_PCA[:, i] = e_vec_PCA[:, i] / np.linalg.norm(e_vec_PCA[:, i])

    # sort according to decreasing value of eigen_values
    index_sorted = np.argsort(-1 * (e_val_PCA))

    e_val_PCA = e_val_PCA[index_sorted]
    e_vec_PCA = e_vec_PCA[:, index_sorted]

    print("size of eigenVector matrix for PCA is :", e_vec_PCA.shape)
    print("size of eigenVector matrix for PCA after trimming to desired components is :",
          e_vec_PCA[:, 0: n_components].shape)

    # return first 'n_components' eigen vectors/ values and overall mean
    return ([e_val_PCA[0: n_components].real, e_vec_PCA[:, 0: n_components].real, mU_overall])


# =========================================================================================
# finding W_lda
# =========================================================================================

def lda(Xprojected, mU_overall, n_classes, n_components, e_vec_pca):
    [row_x, col_x] = Xprojected.shape
    mU_overall_p = np.mat(Xprojected.mean(axis=1))

    n_components = check_num_components(n_components)
    # obtaining scatter matrices
    Sw = np.zeros((row_x, row_x), dtype=np.float32)
    Sb = np.zeros((row_x, row_x), dtype=np.float32)

    for i in range(n_classes):
        Xi = np.dot(e_vec_pca.T, subjects[i].data_subject - mU_overall)

        # mean of each class
        mU_class = np.mat(Xi.mean(axis=1))

        # within class scatter matrix
        Sw += np.dot((Xi - mU_class), (Xi - mU_class).T)
        N_i = Xi.shape[1]

        # between class scatter matrix
        Sb += N_i * np.dot((mU_overall_p - mU_class), (mU_overall_p - mU_class).T)

    [e_val_LDA, e_vec_LDA] = np.linalg.eig(np.dot(np.linalg.inv(Sw), Sb))

    index_sorted = np.argsort(-1 * (e_val_LDA.real))

    e_val_LDA = e_val_LDA[index_sorted]
    e_vec_LDA = e_vec_LDA[:, index_sorted]

    print("size of eigenVector matrix for LDA is :", e_vec_LDA.shape)

    eigenvalues = np.array(e_val_LDA[0: n_components].real, dtype=np.float32, copy=True)
    eigenvectors = np.array(e_vec_LDA[0:, 0: n_components].real, dtype=np.float32, copy=True)

    print("size of eigenVector matrix (after trimming to desired n_components) for LDA is :", e_vec_LDA.shape)

    return ([eigenvalues, eigenvectors])


# =========================================================================================
# check if desired number of components are within the limits. If not set to default value
# =========================================================================================

def check_num_components(n_components):
    if (n_components <= 0) or (n_components > n_classes - 1):
        print("n_components was : ", n_components, " and has been reset to ", n_classes - 1)
        return (n_classes - 1)
    else:
        return (n_components)


# =========================================================================================
# read all files in folder
# image class is assumed to be mentioned as file name itself
# =========================================================================================

# trainingSetDirectory = '/content/drive/My Drive/Photos/yalefacesReducedDataset'
trainingSetDirectory = '/content/drive/My Drive/Photos/trainingWithName'
files = os.listdir(trainingSetDirectory)

img = cv2.imread(trainingSetDirectory + '/' + files[0], cv2.IMREAD_GRAYSCALE)

row, col = img.shape
data = np.reshape(img, (row * col, 1))

print("###shape###", row, col)

# this keeps track of which class number corresponds to which subject
dict = {}

# final value of n_classes serves as number of classes in training set also
n_classes = 0

subjects = []

for file in files:
    # print(".....................................")

    img = cv2.imread(trainingSetDirectory + '/' + file, cv2.IMREAD_GRAYSCALE)

    data = np.reshape(img, (row * col, 1))

    if file[0:9] not in dict:
        dict[file[0:9]] = n_classes

        subjects.append(Subject(data))
        n_classes += 1

    else:
        subject_index = dict[file[0:9]]
        subjects[subject_index].appendData(data)

print("total number of classes in training set is ", n_classes)
X = np.empty([row * col, 1])
X = np.delete(X, 0, 1)

# generate matrix of all images as column vectors
for i in range(n_classes):
    X = np.column_stack((subjects[i].data_subject, X))

[e_val_pca, e_vec_pca, mU_overall] = pca(X, X.shape[1] - n_classes)

[e_val_fld, e_vec_fld] = lda(np.dot(e_vec_pca.T, (X - mU_overall)), mU_overall, n_classes, X.shape[1] - n_classes,
                             e_vec_pca)

# obtain transformation matrix (W) that projects images into C-1 dimensional space
# =========================================================================================
# W_opt = W_pca * W_fld
# =========================================================================================

W = np.dot(e_vec_pca, e_vec_fld)

print("size of transformation matrix W is :", W.shape)

# =========================================================================================
# Testing results
# =========================================================================================
print("====================================================================================")
print("")

print("**************testing results using Fisher Linear Discriminant**************")

# testFileDirectory = '/content/drive/My Drive/Photos/demo'
testFileDirectory = '/./'
testFiles = os.listdir(testFileDirectory)

for testFile in testFiles:
    test = cv2.imread(testFileDirectory + '/' + testFile, cv2.IMREAD_GRAYSCALE)
    test_v = np.reshape(test, (row * col, 1))

    test_v = np.reshape(test, (row * col, 1))
    # print("size of test_image vector is :", test_v.shape)

    # take projection
    testProjection = np.dot(W.T, test_v - mU_overall)
    # print("size of projection of test_image vector is :", testProjection.shape)

    result_class = -1
    minDistance = 0

    # W = e_vec_pca
    for i in range(n_classes):
        for j in range(subjects[i].data_subject.shape[1]):
            sub_Ci_Sj = np.mat(subjects[i].data_subject[:, j]).T

            subProj = np.dot(W.T, sub_Ci_Sj - mU_overall)

            distance = np.linalg.norm(testProjection - subProj)

            # initialize distance for first iteration
            if (i == 0 and j == 0):
                minDistance = distance
                result_class = i

                continue

            if distance < minDistance:
                # print("updating result class to ", i)
                result_class = i
                minDistance = distance

    # =========================================================================================
    # Display results
    # =========================================================================================
    result_subject = list(dict.keys())[list(dict.values()).index(result_class)]
    flagResult = ''
    if (result_subject != testFile[0:9]):
        flagResult = '**FAIL**'

    print("====================================================================================")
    print("")
    print("test_subject : ", testFile, " belongs to : ", result_subject, "      ", flagResult)

print("====================================================================================")
print("")
print("subjects in training set are: ")
print(dict)