# group 8 : Srijeet Chatterjee, Vishal and Ravi Shankar


from skimage import io as skio
from PIL import Image
import numpy as np
import os
import sys
from numpy import linalg as LA
import matplotlib.pyplot as plt

number = 30

'''------------------------Read the data-----------------------------------'''


def read_images(path, sz=None):
    c = 0
    X = []
    y = []

    for dirname, dirnames, filenames in os.walk(path):

        for subdirname in dirnames:
            print(subdirname)

            subject_path = os.path.join(dirname, subdirname)

            for filename in os.listdir(subject_path):
                im = Image.open(os.path.join(subject_path, filename))
                im = im.convert("L")
                image_numpy = np.asarray(np.asarray(im, dtype=np.uint8))
                X.append(image_numpy)
                y.append(c)

            c = c + 1

    return [X, y]


'''--------------------------Load the Data----------------------------------'''

path_train = '/Users/srijeetchatterjee/Desktop/PYTHON_ML/Assignmet_2_ELL784/srijeet/Data/yalefaces/pca_yalefaces/Train_Set/'
files = os.listdir(path_train)
path_test = '/Users/srijeetchatterjee/Desktop/PYTHON_ML/Assignmet_2_ELL784/srijeet/Data/yalefaces/pca_yalefaces/Test_Set/'
files_test = os.listdir(path_test)

[X, y] = read_images(path_train)
[X_test, y_test] = read_images(path_test)

'''----------------- Train and Test Data_Matrix from the List of Images---------------------'''

items = len(X)
[dim1, dim2] = X[0].shape
data_mat = np.empty([0, dim1 * dim2])

for i in range(items):
    [m, n] = (X[i]).shape
    d = m * n
    x_temp = X[i].reshape(1, d)
    data_mat = np.vstack((data_mat, x_temp))

items_test = len(X_test)
[dim1_test, dim2_test] = X_test[0].shape
data_mat_test = np.empty([0, dim1_test * dim2_test])

for i in range(items_test):
    [m, n] = (X_test[i]).shape
    d = m * n
    x_temp_test = X[i].reshape(1, d)
    data_mat_test = np.vstack((data_mat_test, x_temp))

'''-------------------HR - > LR --------------------------------------'''


def process(path, path2, path3, file):
    im = skio.imread(path + file)
    sizeim = [im.shape[0] // 4, im.shape[1] // 4]
    highres = [sizeim[0] * 4, sizeim[1] * 4]

    HRim = im[0:highres[0], 0:highres[1], ...]
    skio.imsave(path2 + file[0:-3] + 'gif', HRim)

    # LRim = scm.imresize(HRim, sizeim, interp='bicubic')
    # skio.imsave(path3 + file[0:-3] + 'gif', LRim)


'''------------------------------ PCA --------------------------------------'''


def PCA(X, y, num):
    items = len(X)
    [dim1, dim2] = X[0].shape
    data_mat = np.empty([0, dim1 * dim2])

    for i in range(items):
        [m, n] = (X[i]).shape
        d = m * n
        x_temp = X[i].reshape(1, d)
        data_mat = np.vstack((data_mat, x_temp))

    no_samples = data_mat.shape[0]
    no_dimension = data_mat.shape[1]

    mu = np.mean(data_mat, axis=0)

    data_mat_mean = data_mat - mu

    if no_samples > no_dimension:
        cov_mat = np.dot(data_mat_mean.T, data_mat_mean)
    if no_samples <= no_dimension:
        cov_mat = np.dot(data_mat_mean, data_mat_mean.T)

    [cov_dim_1, cov_dim_2] = cov_mat.shape

    [eigenvalues, eigenvectors] = LA.eigh(cov_mat)

    eigenvectors = np.dot(data_mat_mean.T, eigenvectors)

    for i in range(no_samples):
        val = LA.norm(eigenvectors[:, i])
        eigenvectors[:, i] = eigenvectors[:, i] / val

    sorting_indexes = np.argsort(-eigenvalues, axis=-1, kind='quicksort', order=None)

    final_eigvectors = np.empty([no_dimension, 0])

    final_eigvals = np.zeros(num)

    for i in range(num):
        index = sorting_indexes[i]
        final_eigvals[i] = eigenvalues[index]
        temp = eigenvectors[:, index]
        final_eigvectors = np.hstack((final_eigvectors, temp.reshape(no_dimension, 1)))

    print(final_eigvals)
    return [final_eigvals, final_eigvectors, mu]


'''---------------Projecting the data into Lower dimension ----------------'''


def projection(W, X, mean):
    nomalized_images_set = np.subtract(X, mean)
    projected_image_matrix = np.dot(nomalized_images_set, W)
    return projected_image_matrix


'''-----------Getting the image back into original dimension ---------------'''


def reconstruct(W, Y, mean):
    converted_in_space = np.dot(Y, W.T)
    nomalized_image = converted_in_space + mean
    return nomalized_image


'''---------------------Eigenfaces model Class-----------------------------'''


class EigenfacesModel():

    def __init__(self, X=None, y=None, num=30):
        self.num = num
        self.projections = []
        self.W = []
        self.mu = []
        self.X = X
        self.y = y
        self.compute(X, y)

    def compute(self, X, y):

        [D, self.W, self.mu] = PCA(X, y, self.num)
        self.y = y

        for xi in X:
            xi_res = xi.reshape(1, -1)
            W = self.W
            mu = self.mu
            n_projection = projection(W, xi_res, mu)
            self.projections.append(n_projection)

    def predict(self, X):

        minDist = sys.float_info.max  # np.finfo('float').max = 1.7976931348623157e+308
        Class = -1

        Q = projection(self.W, X.reshape(1, -1), self.mu)
        q = np.asarray(Q).flatten()

        for i in range(len(self.projections)):

            P = self.projections[i];
            p = np.asarray(P).flatten()

            dist = np.sqrt(np.sum(np.power((p - q), 2)))

            if dist < minDist:
                minDist = dist
                Class = self.y[i]
        return Class


'''--------------------------Fisherfaces Class -----------------------------'''


class FisherfacesModel():

    def __init__(self, X=None, y=None, num=30):
        self.num = 30
        self.projections = []
        self.W = []
        self.mu = []
        self.X = X
        self.y = y
        self.compute(X, y)
        # self.num = num

    def compute(self, X, y):

        [D, self.W, self.mu] = fisherfaces(X, y, self.num)
        self.y = y

        for xi in X:
            self.projections.append(projection(self.W, xi.reshape(1, -1), self.mu))

    def predict(self, X):

        minDist = sys.float_info.max  # np.finfo('float').max = 1.7976931348623157e+308
        Class = -1

        Q = projection(self.W, X.reshape(1, -1), self.mu)
        q = np.asarray(Q).flatten()

        for i in range(len(self.projections)):

            P = self.projections[i];
            p = np.asarray(P).flatten()

            dist = np.sqrt(np.sum(np.power((p - q), 2)))

            if dist < minDist:
                minDist = dist
                Class = self.y[i]
        return Class


'''-----Lets create the model and start predicting----------------------'''

model = EigenfacesModel(X[0:], y[0:], number)

correct_prediction_count = 0

for i in range(len(X_test)):

    predicted_class = model.predict(X_test[i])
    print("Actual Class  = ", y_test[i], "/", " Predicted Class = ", predicted_class)
    if y_test[i] == predicted_class:
        correct_prediction_count = correct_prediction_count + 1

Efficiency = correct_prediction_count / len(X_test)
print(Efficiency * 100)

'''---------------Now lets Project the EigenVectors --------------------------'''

[D_pca, W_pca, mu_pca] = PCA(X, y, number)

# Eigenfaces do not only encode facial features, but also the illumination in the image


path_eigenfaces = '/Users/srijeetchatterjee/Desktop/PYTHON_ML/Assignmet_2_ELL784/srijeet/Eigenfaces_Vectors/'

for i in range(30):
    e = W_pca[:, i].reshape((X[0].shape))
    plt.imshow(e, cmap='jet')
    plt.imsave((path_eigenfaces + str(i) + ".png"), e, cmap='jet')

'''-------------------Visualize the images together----------------------------'''
rows = 4

for i in range(12):
    plt.subplot(rows, 3, i + 1)
    e = W_pca[:, i].reshape((X[0].shape))
    plt.imshow(e, cmap='gray')
    # Image.fromarray(e).show() ii

'''---------------Now project and reconstruct and plot----------------------'''

projected_image = projection(model.W, X[6].reshape(1, -1), model.mu)
reconstructed_image = reconstruct(model.W, projected_image, model.mu)
img = Image.fromarray(reconstructed_image.reshape(X[6].shape))
img.show()

# img.save('my.png')

'''============================FISHER FACES====================================='''


def lda(X, y, num_components=30):
    no_of_features = X.shape[1]

    data_mat = X
    y = np.asarray(y, dtype=np.int64, order=None)

    mean = np.mean(data_mat, axis=0)  # mean of the dataset

    unique, unique_indices, unique_inverse, unique_counts = np.unique(y, return_index=True, return_inverse=True,
                                                                      return_counts=True, axis=None)

    S_w = np.zeros((data_mat.shape[1], data_mat.shape[1]), dtype=np.float32)
    S_b = np.zeros((data_mat.shape[1], data_mat.shape[1]), dtype=np.float32)

    for i in unique:

        temp_mat = np.zeros(shape=(0, data_mat.shape[1]))

        positions = np.where(y == i)[0]

        for j in range(len(positions)):
            index = positions[j]
            temp_mat = np.vstack((temp_mat, data_mat[index, :]))

        temp_mean = temp_mat.mean(axis=0)
        fac = temp_mat - temp_mean
        new_s_w = np.dot((fac).T, (fac))
        S_w = S_w + new_s_w
        S_b_component = (temp_mean - mean).reshape((no_of_features, 1))
        S_b = S_b + (data_mat.shape[0]) * np.dot(S_b_component, S_b_component.T)

    eigenvalues, eigenvectors = LA.eig(LA.inv(S_w) * S_b)

    idx = np.argsort(-eigenvalues.real, axis=-1, kind='quicksort', order=None)
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]
    fin_eigval = eigenvalues[0:num_components].real
    eigenvalues = np.array(fin_eigval, dtype=np.float32, copy=True)
    fin_eigvec = eigenvectors[0:, 0:num_components].real
    eigenvectors = np.array(fin_eigvec, dtype=np.float32, copy=True)

    return [eigenvalues, eigenvectors]


def fisherfaces(X, y, num):
    items = len(X)
    [dim1, dim2] = X[0].shape
    data_mat = np.empty([0, dim1 * dim2])

    for i in range(items):
        [m, n] = (X[i]).shape
        d = m * n
        x_temp = X[i].reshape(1, d)
        data_mat = np.vstack((data_mat, x_temp))

    y = np.asarray(y, dtype=np.int64, order=None)

    c = len(np.unique(y))

    [eigenvalues_pca, eigenvectors_pca, mu_pca] = PCA(X, y, (data_mat.shape[0] - c))

    projected_data_set = np.dot((data_mat - mu_pca), eigenvectors_pca)

    [eigenvalues_lda, eigenvectors_lda] = lda(projected_data_set, y, num)

    eigenvectors = np.dot(eigenvectors_pca, eigenvectors_lda)

    return [eigenvalues_lda, eigenvectors, mu_pca]


# eigenvalues_lda = (no_of_components,)
# eigenvectors =    (no_features_in_original_data_matrix , no_of_components)
# mu_pca = (no_features_in_original_data_matrix,)


'''-----Lets create the model and start predicting----------------------'''

model_1 = FisherfacesModel(X[0:], y[0:], number)

correct_prediction_count = 0

for i in range(len(X_test)):

    predicted_class = model_1.predict(X_test[i])
    print("Actual Class  = ", y_test[i], "/", " Predicted Class = ", predicted_class)
    if y_test[i] == predicted_class:
        correct_prediction_count = correct_prediction_count + 1

Efficiency = correct_prediction_count / len(X_test)
print(Efficiency * 100)

'''----------------------Project the FisherVectors------------------------------'''

[D_fisher, W_fisher, mu_fisher] = fisherfaces(X, y, number)

# not capture illumination as obviously as the Eigenfaces method.
# finds the facial features to discriminate between the persons.

path_fisherfaces = '/Users/srijeetchatterjee/Desktop/PYTHON_ML/Assignmet_2_ELL784/srijeet/Fisherfaces_Vectors/'

for i in range(16):
    e = W_fisher[:, i].reshape((X[0].shape))
    plt.imshow(e, cmap='jet')
    plt.imsave((path_fisherfaces + str(i) + ".png"), e, cmap='jet')

'''-------------------Visualize the images together----------------------------'''

rows = 4

for i in range(12):
    plt.subplot(rows, 3, i + 1)
    e = W_fisher[:, i].reshape((X[0].shape))
    plt.imshow(e, cmap='jet')
    # Image.fromarray(e).show() ii

'''---------------Now project and reconstruct and plot----------------------'''

image_to_be_projected = data_mat[0, :]
mean_of_dataset = mu_fisher
# eigenvec_to_use_for_projection = W[:,0]
projected_image = np.dot(image_to_be_projected - mean_of_dataset, W_fisher)
reconstruct_image = np.dot(projected_image.reshape(1, number), W_fisher.T)
plt.imshow(reconstruct_image.reshape(X[0].shape))

# img =  Image.fromarray(reconstruct_image.reshape(X[0].shape))
# img.show()




