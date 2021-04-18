# Import libraries
from matplotlib import pyplot as plt
from matplotlib.image import imread
from numpy.linalg import inv
import numpy as np
import os

""" Initialize Test and Train Dataset in the ratio of 60-40"""
def init():
    global width, height, number_of_classes, number_of_training_img, pattern_matrix_training, i, j, img, pattern_matrix_testing,count,num_images,correct_pred
    dataset_path = './dataset/'
    dataset_dir = os.listdir(dataset_path)
    width = 92
    height = 112
    number_of_classes = 10
    number_of_training_img = 6
    number_of_testing_img = 4
    total_training_img = number_of_training_img * number_of_classes
    total_testing_img = number_of_testing_img * number_of_classes +4

    """ Face dataset"""
    # to store all the training images in an array
    pattern_matrix_training = np.ndarray(shape=(total_training_img, height * width), dtype=np.float64)
    for i in range(number_of_classes):
        for j in range(number_of_training_img):
            img = plt.imread(dataset_path + 'training1/s' + str(i + 1) + '/' + str(j + 1) + '.pgm')
            # copying images to the training array
            pattern_matrix_training[number_of_training_img * i + j, :] = np.array(img, dtype='float64').flatten()

    # to store all the testing images in an array
    pattern_matrix_testing = np.ndarray(shape=(total_testing_img, height * width), dtype=np.float64)
    for i in range(total_testing_img):
        img = imread(dataset_path + 'test/' + str(i + 1) + '.pgm')
        pattern_matrix_testing[i, :] = np.array(img, dtype='float64').flatten()

    count = 0
    num_images = 0
    correct_pred = 0

""" Apply PCA on the given data and make a database of projected faces"""
def PCA(pattern_matrix_training, number_chosen_components):

    """ Covariance Matrix"""
    mean_face = np.zeros((1,height*width))
    for i in pattern_matrix_training:
        mean_face = np.add(mean_face,i)
    mean_face = np.divide(mean_face, float(pattern_matrix_training.shape[0])).flatten()

    """ Normalization """
    normalised_training_matrix = np.ndarray(shape=(pattern_matrix_training.shape))
    for i in range(pattern_matrix_training.shape[0]):
        normalised_training_matrix[i] = np.subtract(pattern_matrix_training[i], mean_face)

    """ Covariance Matrix"""
    cov_matrix = np.cov(normalised_training_matrix)
    cov_matrix = np.divide(cov_matrix, float(pattern_matrix_training.shape[0]))

    """ Eigenvalues and eigenvectors"""
    eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
    eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]

    """ Feature vector --> choice of k"""
    # Sort the eigen pairs in descending order:
    eig_pairs.sort(reverse=True)
    eigvalues_sort  = [eig_pairs[index][0] for index in range(len(eigenvalues))]
    eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]

    reduced_data = np.array(eigvectors_sort[:number_chosen_components]).transpose()
    """ Eigenfaces """
    # get projected data ---> eigen space
    proj_data = np.dot(pattern_matrix_training.transpose(), reduced_data)
    proj_data = proj_data.transpose()

    """ Weight matrix (important features of each face) """
    wx = np.array([np.dot(proj_data,img) for img in normalised_training_matrix])

    return proj_data, wx


"""Determine Fisher Faces"""
def fisher_faces():
    global projected_data, weight_matrix, i, j, eigvalues_sort, eigvectors_sort, reduced_data, FP, img, mean_face, fig
    # get the projected faces
    number_chosen_components = 30
    projected_data, weight_matrix = PCA(pattern_matrix_training, number_chosen_components)
    """## Mean of each class and global mean"""
    # Each Class Mean
    class_mean = np.zeros((number_of_classes, number_chosen_components))
    # Global Mean
    global_mean = np.zeros((1, number_chosen_components))
    for i in range(number_of_classes):
        xa = weight_matrix[number_of_training_img * i:number_of_training_img * i + number_of_training_img, :]
        for j in xa:
            class_mean[i, :] = np.add(class_mean[i, :], j)
        class_mean[i, :] = np.divide(class_mean[i, :], float(len(xa)))
    for i in weight_matrix:
        global_mean = np.add(global_mean, i)
    global_mean = np.divide(global_mean, float(len(weight_matrix)))
    """## Within class scatter matrix"""
    # normalised within class data
    normalised_wc_proj_sig = np.ndarray(shape=(number_of_classes * number_of_training_img, number_chosen_components),
                                        dtype=np.float64)
    for i in range(number_of_classes):
        for j in range(number_of_training_img):
            normalised_wc_proj_sig[i * number_of_training_img + j, :] = np.subtract(weight_matrix[i * number_of_training_img + j, :],
                                                                          class_mean[i, :])
    sw = np.zeros((number_chosen_components, number_chosen_components))
    for i in range(number_of_classes):
        xa = normalised_wc_proj_sig[number_of_training_img * i:number_of_training_img * i + number_of_training_img, :]
        xa = xa.transpose()
        cov = np.dot(xa, xa.T)
        sw = sw + cov
    """## Between class scatter matrix"""
    normalised_proj_sig = np.ndarray(shape=(number_of_classes * number_of_training_img, number_chosen_components),
                                     dtype=np.float64)
    for i in range(number_of_classes * number_of_training_img):
        normalised_proj_sig[i, :] = np.subtract(weight_matrix[i, :], global_mean)
    sb = np.dot(normalised_proj_sig.T, normalised_proj_sig)
    sb = np.multiply(sb, float(number_of_training_img))
    """## Use the criterion function"""
    J = np.dot(inv(sw), sb)
    """## eigenvalues and eigenvectors"""
    eigenvalues, eigenvectors, = np.linalg.eig(J)
    # get corresponding eigenvectors to eigen values
    # so as to get the eigenvectors at the same corresponding index to eigen values when sorted
    eig_pairs = [(eigenvalues[index], eigenvectors[:, index]) for index in range(len(eigenvalues))]
    # Sort the eigen pairs in descending order:
    eig_pairs.sort(reverse=True)
    eigvalues_sort = [eig_pairs[index][0] for index in range(len(eigenvalues))]
    eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]
    # Choosing the necessary number of principle components from cumulative variance plot
    number_chosen_components = 12
    # Choosing the necessary number of principle components
    max_n_components = number_of_classes - 1
    number_chosen_components = min(max_n_components, number_chosen_components)
    print("k:", number_chosen_components)
    reduced_data = np.array(eigvectors_sort[:number_chosen_components]).transpose()
    """## Fisher Projection(FP)"""
    FP = np.dot(weight_matrix, reduced_data)
    # get projected data ---> eigen space
    proj_data1 = np.dot(pattern_matrix_training.transpose(), FP)
    proj_data1 = proj_data1.transpose()
    mean_face = np.zeros((1, height * width))
    for i in pattern_matrix_training:
        mean_face = np.add(mean_face, i)
    mean_face = np.divide(mean_face, float(len(pattern_matrix_training))).flatten()


def recogniser(img_number):
    global count, highest_min, num_images, correct_pred
    num_images += 1
    unknown_face_vector = pattern_matrix_testing[img_number, :]
    normalised_uface_vector = np.subtract(unknown_face_vector, mean_face)
    count += 1
    PEF = np.dot(projected_data, normalised_uface_vector)
    proj_fisher_test_img = np.dot(reduced_data.T, PEF)
    diff = FP - proj_fisher_test_img
    norms = np.linalg.norm(diff, axis=1)
    index = np.argmin(norms)
    plt.subplot(11, 8, 1 + count)
    set_number = int(img_number / 4)
    # 80% of Max( Min Diff Norm per image)
    t0 = 7000000
    if norms[index] < t0:
        if (index >= (6 * set_number) and index < (6 * (set_number + 1))):
            correct_pred += 1
    else:
        if (img_number >= 40):
            correct_pred += 1
    count += 1
    return correct_pred, num_images

if __name__== '__main__':
    init()
    eigvalues_sort = fisher_faces()
    """ Testing """
    # Testing all the images
    for i in range(len(pattern_matrix_testing)):
        correct_pred, num_images=recogniser(i)
    """ Result"""
    print('Correct predictions: {}/{} = {}%'.format(correct_pred, num_images, correct_pred / num_images * 100.00))










