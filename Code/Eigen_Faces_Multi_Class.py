# Import libraries
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os

num_images = 0
correct_pred = 0
def recogniser(img_number, proj_data, w):
    global num_images, correct_pred
    num_images += 1
    unknown_face_vector = pattern_matrix_testing[img_number,:]
    normalised_uface_vector = np.subtract(unknown_face_vector,mean_face)

    w_unknown = np.dot(proj_data, normalised_uface_vector) # w_known --> projected test face
    diff = w - w_unknown
    norms = np.linalg.norm(diff, axis=1)
    index = np.argmin(norms)
    set_number = int(img_number/4)
    t0 = 15000000
    if norms[index] < t0:
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

if __name__ == '__main__':
    dataset_path = './dataset/'
    dataset_dir = os.listdir(dataset_path)
    width = 92
    height = 112
    Number_of_classes = 10
    Number_of_training_img = 6
    Number_of_testing_img = 4
    total_training_img = Number_of_training_img * Number_of_classes
    total_testing_img = Number_of_testing_img * Number_of_classes + 4

    """ Face dataset"""
    # to store all the training images in an array
    pattern_matrix_training = np.ndarray(shape=(total_training_img, height*width), dtype=np.float64)
    for i in range(total_training_img):
        img = plt.imread(dataset_path + 'training/'+str(i+1)+'.pgm')
        pattern_matrix_training[i, :] = np.array(img, dtype='float64').flatten()

    # to store all the testing images in an array
    pattern_matrix_testing = np.ndarray(shape=(total_testing_img, height*width), dtype=np.float64)
    for i in range(total_testing_img):
        img = imread(dataset_path + 'test/'+str(i+1)+'.pgm')
        pattern_matrix_testing[i, :] = np.array(img, dtype='float64').flatten()

    """ Mean Calculation """
    mean_face = np.zeros((1, height*width))
    for i in pattern_matrix_training:
        mean_face = np.add(mean_face, i)
    mean_face = np.divide(mean_face, float(len(pattern_matrix_training))).flatten()

    """ Normalization """
    # normalizing training set
    normalised_pattern_matrix_training = np.ndarray(shape=(len(pattern_matrix_training), height*width))
    for i in range(len(pattern_matrix_training)):
        normalised_pattern_matrix_training[i] = np.subtract(pattern_matrix_training[i], mean_face)

    """ Covariance Matrix"""
    cov_matrix = np.cov(normalised_pattern_matrix_training)
    cov_matrix = np.divide(cov_matrix, total_training_img)

    """ Eigenvalues and eigenvectors"""
    eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)

    """ Feature vector --> choice of k"""
    eig_pairs = [(eigenvalues[index], eigenvectors[:, index]) for index in range(len(eigenvalues))]
    # Sort the eigen pairs in descending order:
    eig_pairs.sort(reverse=True)
    eigvalues_sort = [eig_pairs[index][0] for index in range(len(eigenvalues))]
    eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]

    # x-axis for number of principal components kept
    num_comp = range(1, len(eigvalues_sort)+1)

    # Choosing the necessary number of principle components
    number_chosen_components = 30
    print("k:", number_chosen_components)
    reduced_data = np.array(eigvectors_sort[:number_chosen_components]).transpose()

    """ Eigenfaces """
    # get projected data ---> eigen space
    proj_data = np.dot(pattern_matrix_training.transpose(), reduced_data)
    proj_data = proj_data.transpose()

    """ Weight matrix (important features of each face) """
    w = np.array([np.dot(proj_data, img) for img in normalised_pattern_matrix_training])

    """ Testing """
    # Testing all the images
    for i in range(len(pattern_matrix_testing)):
        recogniser(i, proj_data, w)

    """ Result"""
    print('Correct predictions: {}/{} = {}%'.format(correct_pred, num_images, correct_pred/num_images*100.00))

