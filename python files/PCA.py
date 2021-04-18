from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
import cv2 as cv
import os

# dataset_path = '.'
# dataset_dir = os.listdir(dataset_path)
# img1 = cv.imread(dataset_path + 'training/' + str(1) + '.pgm')
img1 = cv.imread('./training/' + str(1) + '.pgm')
a = img1.shape
print(a)
width = a[1]
height = a[0]
print(width, height)

print('Train Images:')

# to store all the training images in an array
pattern_matrix_training = np.ndarray(shape=(18, height * width), dtype=np.float64) # please change to req num

for i in range(18): # Taken 3 classes for now. please change it to required number
    # img = plt.imread(dataset_path + 'training/' + str(i + 1) + '.pgm')
    img = plt.imread('./training/' + str(i + 1) + '.pgm')
    # copying images to the training array
    pattern_matrix_training[i, :] = np.array(img, dtype='float64').flatten()


print('Test Images:')
pattern_matrix_testing = np.ndarray(shape=(12, height * width), dtype=np.float64) # please change to req num

for i in range(12): #Taken 3 classes for testing. please change it to req number
    # img = imread(dataset_path + 'test/' + str(i + 1) + '.pgm')
    img = imread('./test/' + str(i + 1) + '.pgm')
    pattern_matrix_testing[i, :] = np.array(img, dtype='float64').flatten()


mean_face = np.zeros((1, height * width))

for i in pattern_matrix_training:
    mean_face = np.add(mean_face, i)

mean_face = np.divide(mean_face, float(len(pattern_matrix_training))).flatten()
print(mean_face)


# normalized training set
normalised_pattern_matrix_training = np.ndarray(shape=(len(pattern_matrix_training), height * width))

for i in range(len(pattern_matrix_training)):
    normalised_pattern_matrix_training[i] = np.subtract(pattern_matrix_training[i], mean_face)

for i in range(len(pattern_matrix_training)):
    img = normalised_pattern_matrix_training[i].reshape(height, width)

cov_matrix = np.cov(normalised_pattern_matrix_training)
cov_matrix = np.divide(cov_matrix,18) # please change 18 to the required number(num of train images)

print('Covariance matrix of X:')
print(cov_matrix)

eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
print('Eigenvectors of Cov(X):')
print(eigenvectors)
print('Eigenvalues of Cov(X):', eigenvalues)

# get corresponding eigenvectors to eigen values
# so as to get the eigenvectors at the same corresponding index to eigen values when sorted
eig_pairs = [(eigenvalues[index], eigenvectors[:, index]) for index in range(len(eigenvalues))]

# Sort the eigen pairs in descending order:
eig_pairs.sort(reverse=True)
eigvalues_sort = [eig_pairs[index][0] for index in range(len(eigenvalues))]
eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]

# Find cumulative variance of each principle component
var_comp_sum = np.cumsum(eigvalues_sort) / sum(eigvalues_sort)

# Show cumulative proportion of varaince with respect to components
print("Cumulative proportion of variance explained vector:", var_comp_sum)

# x-axis for number of principal components kept
num_comp = range(1, len(eigvalues_sort) + 1)


print('Number of eigen vectors:', len(eigvalues_sort))

# Choosing the necessary number of principle components
number_chosen_components = 30  # 30// Remove hard coding
print("k:", number_chosen_components)
reduced_data = np.array(eigvectors_sort[:number_chosen_components]).transpose()

# get projected data ---> eigen space

proj_data = np.dot(pattern_matrix_training.transpose(), reduced_data)
proj_data = proj_data.transpose()

# plotting of eigen faces --> the information retained after applying lossing transformation
# for i in range(proj_data.shape[0]):
#     img = proj_data[i].reshape(height, width)
#     plt.subplot(10, 6, 1 + i)
#     plt.imshow(img, cmap='jet')
#     plt.subplots_adjust(right=1.2, top=2.5)
#     plt.tick_params(labelleft=False, labelbottom=False, bottom=False, top=False, right=False, left=False, which='both')
# plt.show()

proj_data.shape

print(normalised_pattern_matrix_training.shape)
print(proj_data.shape)

w = np.array([np.dot(proj_data, img) for img in normalised_pattern_matrix_training])
w.shape

# Testing all the images

count = 0
num_images = 0
correct_pred = 0


def recogniser(img_number, proj_data, w):
    global count, highest_min, num_images, correct_pred

    num_images += 1
    unknown_face_vector = pattern_matrix_testing[img_number, :]
    normalised_uface_vector = np.subtract(unknown_face_vector, mean_face)

    # plt.subplot(11, 8, 1 + count)
    # plt.imshow(unknown_face_vector.reshape(height, width), cmap='gray')
    # plt.title('Input:' + str(img_number + 1))
    # plt.subplots_adjust(right=1.2, top=2.5)
    # plt.tick_params(labelleft=False, labelbottom=False, bottom=False, top=False, right=False, left=False, which='both')
    count += 1

    w_unknown = np.dot(proj_data, normalised_uface_vector)  # w_known --> projected test face
    diff = w - w_unknown
    norms = np.linalg.norm(diff, axis=1)
    # print('norms::', norms[0])

    index = np.argmin(norms)

    # print('index::', index)
    # print('min::', np.min(norms))

    # plt.subplot(11, 8, 1 + count)

    set_number = int(img_number / 4)  # ???
    # print('set what', set_number)
    #     print(set_number)

    t0 = 25000000

    # if(img_number>=40):
    #     print(norms[index])

    if norms[index] < t0:
        # 1st Subject > 0 and < 6
        # 2nd Subject > 6 and 12
        if (index >= (6 * set_number) and index < (6 * (set_number + 1))):
            plt.title('Matched with:' + str(index + 1), color='g')
            plt.imshow(pattern_matrix_training[index, :].reshape(height, width), cmap='gray')
            correct_pred += 1
        else:
            plt.title('Matched with:' + str(index + 1), color='r')
            plt.imshow(pattern_matrix_training[index, :].reshape(height, width), cmap='gray')
    else:
        if (img_number >= 40):
            plt.title('Unknown face!', color='g')
            correct_pred += 1
        else:
            plt.title('Unknown face!', color='r')
    plt.subplots_adjust(right=1.2, top=2.5)
    plt.tick_params(labelleft=False, labelbottom=False, bottom=False, top=False, right=False, left=False, which='both')
    count += 1


fig = plt.figure(figsize=(10, 10))
for i in range(len(pattern_matrix_testing)):
    recogniser(i, proj_data, w)

# plt.show()

print('Correct predictions: {}/{} = {}%'.format(correct_pred, num_images, correct_pred / num_images * 100.00))

accuracy = np.zeros(len(eigvalues_sort))


def tester(img_number, proj_data, w, num_images, correct_pred):
    num_images += 1
    unknown_face_vector = pattern_matrix_testing[img_number, :]
    normalised_uface_vector = np.subtract(unknown_face_vector, mean_face)

    w_unknown = np.dot(proj_data, normalised_uface_vector)
    diff = w - w_unknown
    norms = np.linalg.norm(diff, axis=1)
    index = np.argmin(norms)

    set_number = int(img_number / 4)

    t0 = 25000000

    if norms[index] < t0:
        if (index >= (6 * set_number) and index < (6 * (set_number + 1))):
            correct_pred += 1
    else:
        if (img_number >= 40):
            correct_pred += 1

    return num_images, correct_pred


def calculate(k):
    #     print("k:",k)
    reduced_data = np.array(eigvectors_sort[:k]).transpose()

    proj_data = np.dot(pattern_matrix_training.transpose(), reduced_data)
    proj_data = proj_data.transpose()

    w = np.array([np.dot(proj_data, img) for img in normalised_pattern_matrix_training])

    num_images = 0
    correct_pred = 0

    for i in range(len(pattern_matrix_testing)):
        num_images, correct_pred = tester(i, proj_data, w, num_images, correct_pred)

    accuracy[k] = correct_pred / num_images * 100.00


# print('Total Number of eigenvectors:',len(eigvalues_sort))
for i in range(1, len(eigvalues_sort)):
    calculate(i)

# fig, axi = plt.subplots()
# axi.plot(np.arange(len(eigvalues_sort)), accuracy, 'b')
# axi.set_xlabel('Number of eigen values')
# axi.set_ylabel('Accuracy')
# axi.set_title('Accuracy vs. k-value')

len(pattern_matrix_training)
pattern_matrix_training = np.array(pattern_matrix_training)
print(pattern_matrix_training)
y = []
for i in range(10):
    for j in range(6):
        y.append(i)
print(y)
y = np.array(y)
print(y)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_r = pca.fit(pattern_matrix_training).transform(pattern_matrix_training)
print(len(pattern_matrix_training), len(pattern_matrix_training[0]))

target_names = []
for i in range(10):
    target_names.append(str(i))
print(target_names)
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

# plt.figure()
# colors = ['navy', 'turquoise']
# lw = 2
# t = []
# for i in range(60):
#     t.append(i)
# for i, target_name in zip(t, target_names):
#     plt.scatter(X_r[y == i, 0], X_r[y == i, 1], alpha=.8, lw=lw,
#                 label=target_name)
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.title('Training Images')
# plt.show()