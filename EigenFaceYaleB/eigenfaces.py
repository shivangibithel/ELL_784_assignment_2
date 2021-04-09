import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image
import scipy.io
#loading training images
path = './train/'
files = os.listdir(path)
n= len(files)
print(n)
images = np.empty(shape=(400,n),dtype='int8')
i = 0
for filename in files:
    temp1 = cv2.imread(path+filename)
    temp = cv2.cvtColor(temp1,cv2.COLOR_RGB2GRAY)
    temp = cv2.resize(temp,(20,20))
    img = np.array(temp,dtype='int8').flatten()
    images[:,i] = img[:]
    i = i+1
train_num = i
mean = images.mean(axis = 1)
for j in range(train_num):
    images[:,j] = images[:,j]-mean

cov = np.matmul(images,images.T)/train_num
print(cov.shape)

eigvals,eigvecs = np.linalg.eigh(cov)
norm = np.linalg.norm(eigvecs, axis=0)
eigvecs = eigvecs / norm
k= 300 #considering only k eigen values
p = eigvals.argsort()[::-1]
#sorting eigen values and vectors
eigvals = eigvals[p]
eigvecs = eigvecs[:, p]
A = eigvecs[:, 0:k]
#
Y = np.matmul(A.T,images)
# #SVD function
u,s,v = np.linalg.svd(cov)
print(u.shape, s.shape , v.shape)
###################prediction######################################################3
# Reading test image as an input, converting into 100*100
test_path = './test/yaleB04_P00A+010E-20.pgm'
test = np.array(cv2.imread(test_path))
test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
test = cv2.resize(test, (20,20))

img = test.reshape(400,1)

# Substracting mean
img = img - mean
img = img.T

#calculating & comparing the euclidian distance of all projected trained images from the projected test image###############

# Dot product of test image and U matrix
test_x = np.empty(shape=(u.shape[0], u.shape[1]), dtype=np.int8)
print(test_x.shape)

for col in range(u.shape[1]):
    test_x[:, col] = img[:, 0] * u[:, col]

dot_test = np.array(test_x, dtype='int8').flatten()



# Dot product of all the images and U matrix
dot_train = np.empty(shape=(u.shape[0] * u.shape[1], u.shape[1]), dtype='int8')
temp = np.empty(shape=(u.shape[0], u.shape[1]), dtype=np.int8)

for i in range(images.shape[1]):
    for c in range(u.shape[1]):
        temp[:, c] = images[:, i] * u[:, c]

    tempF = np.array(temp, dtype='int8').flatten()
    dot_train[:, i] = tempF[:]


# Substracting Two dot products
sub = np.empty(shape=(u.shape[0] * u.shape[1], u.shape[1]))

for col in range(u.shape[1]):
    sub[:, col] = dot_train[:, col] - dot_test[:]



# Finding norm of all the columns
answer = np.empty(shape=(u.shape[1],))

for c in range(sub.shape[1]):
    answer[c] = np.linalg.norm(sub[:, c])



# print(answer)



# Sorting answer array and retriving first element which will be minimum from all
temp_ans = np.empty(shape=(u.shape[1],))
temp = np.copy(answer)

temp.sort()
check = temp[0]
# print(check)


index = 0

for i in range(answer.shape[0]):
    if check == answer[i]:
        index = i
        break


# Checking for corresponding image for minimum answer
i = 0
# print(index)
for filename in files:
    if index == i:
        print("The predicted face is: ",filename)
        # prediction = cv2.imread(filename)
        break
    else:
        i = i + 1
    
fig = plt.figure(figsize=(10,7))
fig.add_subplot(1,2,1)
X = plt.imread(path+filename)
plt.imshow(X)
plt.title('predicted')
fig.add_subplot(1,2,2)
Tru_img = plt.imread(test_path)
plt.imshow(Tru_img)
plt.title('test img')
plt.show()













