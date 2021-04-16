"""# Plot Decision Boundary PCA"""

len(pattern_matrix_training)
pattern_matrix_training = np.array(pattern_matrix_training)
print(pattern_matrix_training)
y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
y = np.array(y)
print(y)

pca = PCA(n_components=2)
X_r = pca.fit(pattern_matrix_training).transform(pattern_matrix_training)
print(len(pattern_matrix_training), len(pattern_matrix_training[0]))

# Commented out IPython magic to ensure Python compatibility.
target_names = ['subject0', 'subject1']
# print('explained variance ratio (first two components): %s'
#       % str(pca.explained_variance_ratio_))

y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
y = np.array(y)
print(y)

# plt.figure()
colors = ['navy', 'turquoise']
lw = 2
for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Determine data boundary')
plt.show()

""" Plot decision boundary LDA """

print("Number of features:", len(pattern_matrix_training[0]))

#### Labels ######
x1 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
x1 = np.array(x1)
print(x1)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda_object = lda.fit_transform(pattern_matrix_training, x1)

plt.figure()
colors = ['navy', 'turquoise']
target_names = ['subject0', 'subject1']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(lda_object[x1 == i, 0], lda_object[x1 == i, 0], alpha=.8, color=color,
                label=target_name)

plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of ATT dataset')
plt.show()

"""# K-means"""

# Importing required modules
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

# Load Data
data = pattern_matrix_training
pca = PCA(2)

# Transform the data
df = pca.fit_transform(data)

df.shape

# Import required module
from sklearn.cluster import KMeans

# Initialize the class object
kmeans = KMeans(n_clusters=2)

# predict the labels of clusters.
label = kmeans.fit_predict(df)

print(label)

"""Plot for Label 0 """

import matplotlib.pyplot as plt

# filter rows of original data
filtered_label0 = df[label == 0]

# plotting the results
plt.scatter(filtered_label0[:, 0], filtered_label0[:, 1])
plt.show()

"""Plot for label 1"""

import matplotlib.pyplot as plt

# filter rows of original data
filtered_label0 = df[label == 1]

# plotting the results
plt.scatter(filtered_label0[:, 0], filtered_label0[:, 1])
plt.show()

"""Plot for label 0 and label 1"""

# filter rows of original data
filtered_label0 = df[label == 0]
filtered_label1 = df[label == 1]

# Plotting the results
plt.scatter(filtered_label0[:, 0], filtered_label0[:, 1], color='red')
plt.scatter(filtered_label1[:, 0], filtered_label1[:, 1], color='black')
plt.show()

"""Plot with centroids"""

# Getting the Centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(label)

# plotting the results:

for i in u_labels:
    plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
plt.legend()
plt.show()

""" Plotting of Accuracy vs k-value for eigenvectors """
# accuracy = np.zeros(len(eigvalues_sort))
# def tester(img_number,proj_data,w,num_images,correct_pred):
#
#     num_images += 1
#     unknown_face_vector = pattern_matrix_testing[img_number,:]
#     normalised_uface_vector = np.subtract(unknown_face_vector,mean_face)
#
#     w_unknown = np.dot(proj_data, normalised_uface_vector)
#     diff  = w - w_unknown
#     norms = np.linalg.norm(diff, axis=1)
#     index = np.argmin(norms)
#     set_number = int(img_number/4)
#     if(index>=(6*set_number) and index<(6*(set_number+1))):
#       correct_pred += 1
#     return num_images, correct_pred

# def calculate(k):
#     reduced_data = np.array(eigvectors_sort[:k]).transpose()
#
#     proj_data = np.dot(pattern_matrix_training.transpose(),reduced_data)
#     proj_data = proj_data.transpose()
#
#     w = np.array([np.dot(proj_data,img) for img in normalised_pattern_matrix_training])
#
#     num_images = 0
#     correct_pred = 0
#     for i in range(len(pattern_matrix_testing)):
#         num_images, correct_pred = tester(i, proj_data, w, num_images, correct_pred)
#     accuracy[k] = correct_pred/num_images*100.00
#
# print('Total Number of eigenvectors:',len(eigvalues_sort))
# for i in range(1, len(eigvalues_sort)):
#     calculate(i)