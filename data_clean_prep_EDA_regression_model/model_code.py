# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:35:07 2022

@author: leube
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# lets get some pictures and experiment with set up, arrangement in np.arrays etc.

import PIL
print('Pillow Version:', PIL.__version__)

from PIL import Image
# Open the image form working directory
image1 = Image.open(r"C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\gun_experiment.jpg")
# summarize some details about the image
print(image1.format)
print(image1.size)
print(image1.mode)
# show the image
image1.show()


# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot
# load image as pixel array
image = image.imread(r"C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\gun_experiment.jpg")
# summarize shape of the pixel array
print(image.dtype)
print(image.shape)
# display the array of pixels as an image
pyplot.imshow(image)
pyplot.show()


# converting image to array
from numpy import asarray

data = asarray(image)
data.shape


# reshape image

pic = np.array(Image.open(r"C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\gun_experiment.jpg").resize((100,100)))

pic.shape
pic


# flattening the image


import numpy as np
import pandas as pd
from skimage import io
from PIL import Image
import matplotlib.pylab as plt
from numpy import array
from sys import getsizeof

#Using Flatten function on array 1 to convert the multi-dimensional 
# array to 1-D array
array2 = pic.flatten()
pic.shape
#Memory occupied by array 2
size2 = getsizeof(array2)
#displaying the 1-D array
print(array2)
array2.shape


fig, ax = plt.subplots()
ax.imshow(array2.reshape((200,200,3)), cmap=plt.cm.bone, interpolation='nearest')

# lets try ith webp pictures
image = Image.open(r"C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\image_1.webp")
# summarize some details about the image
print(image.format)
print(image.size)
print(image.mode)
# show the image
image.show()

# next test
image
image = image.imread(r"C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\image_1.webp")

#webp = ("C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\image_1.webp")

# summarize shape of the pixel array
print(image.dtype)
print(image.shape)
# display the array of pixels as an image
pyplot.imshow(image)
pyplot.show()
'''
result: webp pictures do not need to be image.imread converted

'''

# lets reshape
pic = np.array(Image.open(r"C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\gun_experiment.jpg").resize((100,100)))



picweb = np.array(Image.open(r"C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\image_1.webp").resize((46,46)))

picweb.shape
pic

# lets flatten and then reshape and plot

# lets reshape and plot the immage
array2 = picweb.flatten()
picweb.shape
#Memory occupied by array 2
size2 = getsizeof(array2)
#displaying the 1-D array
print(array2)
array2.shape
fig, ax = plt.subplots()
ax.imshow(array2.reshape((46,46,3)), cmap=plt.cm.bone, interpolation='nearest')

## all of the experiments succesful, lets get some images

# import images one by one and put array into a list


list_all_arrays = []

for x in range(1,41):
    linkstring=r"C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\image_" + str(x)+ ".jpg"
    print(linkstring)
    
    
    
    #linkstring = f"C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\image_{x}.jpg"
    #link = r"C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\image_{x}.jpg"
    #image = image.imread(linkstring)
    # summarize shape of the pixel array
    #print(image.dtype)
    #print(image.shape)
    # display the array of pixels as an image
    #pyplot.imshow(image)
    #pyplot.show()
    
    # resize
    
    pic = np.array(Image.open(linkstring).resize((80,80)))
    
    # flatten
    
    array2 = pic.flatten()

    #Memory occupied by array 2

    #displaying the 1-D array
    print(array2)
    list_all_arrays.append(array2)

list_all_arrays

# make a list containing of the list turned arrays

lists_in_list = [list(x) for x in list_all_arrays]
print(lists_in_list)

for x in range(len(lists_in_list)):
    print(lists_in_list[x])
    break

finished_array = np.array(lists_in_list)
finished_array


# checks

# check original shape of them to know how to reshape for image picuring




# check

for x in finished_array:
    fig, ax = plt.subplots()
    ax.imshow(x.reshape((80,80,3)), cmap=plt.cm.bone, interpolation='nearest')
    plt.show()
    print(x)


for x in finished_array:
    print(x.shape)
    
    
fig, ax = plt.subplots()
ax.imshow(finished_array[25].reshape((80,80,3)), cmap=plt.cm.bone, interpolation='nearest')
plt.show()


# all reshaping and flattening succesfull, lets start with the model
X = finished_array

# make a list that contains label for each of the 30 pictures, we know tht the first 10 pictures contained weapons
list1 = [1 for x in range(1,11)]

list0 = [0 for x in range (1,21)]
ylist = list1 + list0
Y = np.array(ylist)

# create train test splits of data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=20)


import tensorflow as tf


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(110, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(80, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(50, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)

model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(X_train,y_train, epochs=70)

val_loss, val_acc = model.evaluate(X_test, y_test)


for x in X_test:
    fig, ax = plt.subplots()
    ax.imshow(x.reshape((80,80,3)), cmap=plt.cm.bone, interpolation='nearest')
    plt.show()
    print(x)
    
    
y_test

# recieving not very optimal results, the model is overfitting a lot, lets try both trying different
# hyper parameters and also especially scaling the pixels first 


arraydf = pd.DataFrame(finished_array)

arraydftr = arraydf.transpose()

for x in list(arraydftr.columns):
    mini = arraydftr[x].min()
    maxi = arraydftr[x].max()
    arraydftr[x] = (arraydftr[x]-mini)/(maxi-mini)

finardf = arraydftr.transpose()

finardf

# add new values to Y
Yl = list(Y)
for x in range(10):
    Yl.append(1)
    
Y = np.array(Yl)

len(Y)



# lets try it again with normalized data

X_train, X_test, y_train, y_test = train_test_split(finardf, Y, test_size=0.20, random_state=20)


X_train = np.asarray(X_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(110, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(80, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(50, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)


model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(X_train,y_train, epochs=35)


val_loss, val_acc = model.evaluate(X_test, y_test)

'''
val_loss, val_acc = model.evaluate(X_test, y_test)
1/1 [==============================] - 0s 144ms/step - loss: 6.5369 - accuracy: 0.5000

'''

# lets try other activation functions


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(110, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(80, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(50, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.sigmoid))

tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)


model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(X_train,y_train, epochs=120)


val_loss, val_acc = model.evaluate(X_test, y_test)

# lets try pca?
def standardize(x):
    #This function takes as input a numpy array and returns a stadardized numpy array of the same dimensions.
    #Sample input: np.array([[1,2,3],[4,5,6]])
    #Sample output: np.array([[-1,-1,-1],[1,1,1]])
    
    # Your code here:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    output = [(pix-mean)/std for pix in list(x)]
    return np.array(output)

finished_array
finished_array_st = np.apply_along_axis(standardize,axis=0, arr=finished_array)

from sklearn.decomposition import PCA

Xpca = finished_array_st


pca = PCA(n_components=20)
components = pca.fit_transform(Xpca)

components.shape


# reapply model

X_train, X_test, y_train, y_test = train_test_split(components, Y, test_size=0.20, random_state=20)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(110, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(80, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(50, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.sigmoid))

tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)


model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(X_train,y_train, epochs=90)



val_loss, val_acc = model.evaluate(X_test, y_test)

# lets try other number of components


Xpca = finished_array_st


pca = PCA(n_components=7)
components2 = pca.fit_transform(Xpca)

components2.shape

# model

X_train, X_test, y_train, y_test = train_test_split(components2, Y, test_size=0.20, random_state=20)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(110, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(80, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(50, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.sigmoid))

tf.keras.optimizers.Adam(
    learning_rate=0.01,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)


model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(X_train,y_train, epochs=90)



val_loss, val_acc = model.evaluate(X_test, y_test)


# lets add new pictures








list_all_arrays = []

for x in range(1,48):
    linkstring=r"C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\image_" + str(x)+ ".jpg"
    print(linkstring)
    
    # resize
    
    pic = np.array(Image.open(linkstring).resize((80,80)))
    
    # flatten
    array2 = pic.flatten()

    #Memory occupied by array 2

    #displaying the 1-D array
    print(array2)
    list_all_arrays.append(array2)

len(list_all_arrays)

# make a list containing of the list turned arrays

lists_in_list = [list(x) for x in list_all_arrays]
print(lists_in_list)

for x in range(len(lists_in_list)):
    print(lists_in_list[x])
    break
lists_in_list = [list(x) for x in list_all_arrays]
print(lists_in_list)
finished_array = np.array(lists_in_list)
finished_array


# checks

# check original shape of them to know how to reshape for image picuring




# check

for x in finished_array:
    fig, ax = plt.subplots()
    ax.imshow(x.reshape((80,80,3)), cmap=plt.cm.bone, interpolation='nearest')
    plt.show()
    print(x)


for x in finished_array:
    print(x.shape)
    
    
fig, ax = plt.subplots()
ax.imshow(finished_array[25].reshape((80,80,3)), cmap=plt.cm.bone, interpolation='nearest')
plt.show()


# all reshaping and flattening succesfull, lets start with the model

# make a list that contains label for each of the 30 pictures, we know tht the first 10 pictures contained weapons
list1 = [1 for x in range(1,11)]

list0 = [0 for x in range (1,21)]
list12 = [1 for x in range(1,18)]
ylist = list1 + list0 + list12
Y = np.array(ylist)
len(Y)

finished_array.shape
finished_array_st = np.apply_along_axis(standardize,axis=0, arr=finished_array)


X = finished_array_st
len(finished_array_st)

Xpca = finished_array_st
finished_array_st.shape

pca = PCA(n_components=20)
componentsnew = pca.fit_transform(Xpca)

componentsnew.shape

# run model again

X_train, X_test, y_train, y_test = train_test_split(componentsnew, Y, test_size=0.20, random_state=20)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(110, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(80, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(50, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.sigmoid))

tf.keras.optimizers.Adam(
    learning_rate=0.01,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)


model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(X_train,y_train, epochs=50)



val_loss, val_acc = model.evaluate(X_test, y_test)


# lets systematically try out different parameters

# lower number of layers, only sigmoid


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.sigmoid))

tf.keras.optimizers.Adam(
    learning_rate=0.01,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)


model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(X_train,y_train, epochs=50)



val_loss, val_acc = model.evaluate(X_test, y_test)

# only relu


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.relu))

tf.keras.optimizers.Adam(
    learning_rate=0.01,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)


model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(X_train,y_train, epochs=60)



val_loss, val_acc = model.evaluate(X_test, y_test)
'''
val_loss, val_acc = model.evaluate(X_test, y_test)
1/1 [==============================] - 0s 100ms/step - loss: 0.2773 - accuracy: 0.9000


why does it change if I run the excat same code again??

'''
# lets try higher learning rate


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.relu))

tf.keras.optimizers.Adam(
    learning_rate=0.03,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)


model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(X_train,y_train, epochs=30)



val_loss, val_acc = model.evaluate(X_test, y_test)


# more epochs

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.relu))

tf.keras.optimizers.Adam(
    learning_rate=0.03,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)


model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(X_train,y_train, epochs=60)



val_loss, val_acc = model.evaluate(X_test, y_test)


# not an option, lets try less epochs

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.relu))

tf.keras.optimizers.Adam(
    learning_rate=0.01,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)


model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(X_train,y_train, epochs=24)



val_loss, val_acc = model.evaluate(X_test, y_test)

'''
this one seems promising
1/1 [==============================] - 0s 112ms/step - loss: 0.2079 - accuracy: 1.0000

'''

# lets save this one for later use

model.save('best_weapon_model_wed_1930.model')
new_model = tf.keras.models.load_model('best_weapon_model_wed_1930.model')


# lets make some predictions

# loading test image

testimage= np.array(Image.open(r"C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\image_48.jpg").resize((80,80)))
testarray = testimage.flatten()

testarray

# standardize
test_array_st = np.apply_along_axis(standardize,axis=0, arr=testarray)
test_array_st.shape
test_array_st
# pca reduction
Xpca = test_array_st.reshape(1,19200)


pca = PCA(n_components=20)
componentstest = pca.fit_transform(Xpca)

componentstest.shape

# something is wrong here with pca, ask ksneia tomorrow how to properky use pca in this case


# checking label


predictions = new_model.predict(componentstest)
score = tf.nn.softmax(predictions[0])




print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


## try the good model with out pca

finished_array_st.shape

X_train, X_test, y_train, y_test = train_test_split(finished_array_st, Y, test_size=0.20, random_state=20)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.relu))

tf.keras.optimizers.Adam(
    learning_rate=0.01,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)


model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(X_train,y_train, epochs=26)



val_loss, val_acc = model.evaluate(X_test, y_test)



# lets try it with a dataframe I feel like something is wrong here with the arrays

finstdf = pd.DataFrame(finished_array_st)
finstdf['labels'] = Y
finstdf['labels']


Xdf = finstdf.drop('labels', axis=1)
ydf = finstdf['labels']

X_train, X_test, y_train, y_test = train_test_split(Xdf, ydf, test_size=0.20, random_state=20)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.relu))

tf.keras.optimizers.Adam(
    learning_rate=0.01,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)


model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(X_train,y_train, epochs=26)



val_loss, val_acc = model.evaluate(X_test, y_test)


# okay so has nothing to do with dataframe, just have to figure out if pca is done correctly

X_train, X_test, y_train, y_test = train_test_split(finished_array_st, Y, test_size=0.20, random_state=50)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.relu))

tf.keras.optimizers.Adam(
    learning_rate=0.01,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)


model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(X_train,y_train, epochs=26)



val_loss, val_acc = model.evaluate(X_test, y_test)

predictions = new_model.predict(test_array_st)
score = tf.nn.softmax(predictions[0])




print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))






# maybe its not the model, its the way I feed my data, lets explore a few ways how we can work with pixels

# lets work with a wepon picture and a random other picture



im = plt.imread(r"C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\image_1.jpg")

def to_grayscale(im, weights = np.c_[0.2989, 0.5870, 0.1140]):
    """
    Transforms a colour image to a greyscale image by
    taking the mean of the RGB values, weighted
    by the matrix weights
    """
    tile = np.tile(weights, reps=(im.shape[0],im.shape[1],1))
    return np.sum(tile * im, axis=2)

def plti(im, h=8, **kwargs):
    """
    Helper function to plot an image.
    """
    y = im.shape[0]
    x = im.shape[1]
    w = (y/x) * h
    plt.figure(figsize=(w,h))
    plt.imshow(im, interpolation="none", **kwargs)
    plt.axis('off')

plti(im)


def simple_threshold(im, threshold=128):
    return ((im > threshold) * 255).astype("uint8")

thresholds = [100,120,128,138,150]

fig, axs = plt.subplots(nrows=1, ncols=len(thresholds), figsize=(20,5));
gray_im = to_grayscale(im)

plti(gray_im, cmap='Greys')

def otsu_threshold(im):

    pixel_counts = [np.sum(im == i) for i in range(256)]

    s_max = (0,-10)
    ss = []
    for threshold in range(256):

        # update
        w_0 = sum(pixel_counts[:threshold])
        w_1 = sum(pixel_counts[threshold:])

        mu_0 = sum([i * pixel_counts[i] for i in range(0,threshold)]) / w_0 if w_0 > 0 else 0       
        mu_1 = sum([i * pixel_counts[i] for i in range(threshold, 256)]) / w_1 if w_1 > 0 else 0

        # calculate 
        s = w_0 * w_1 * (mu_0 - mu_1) ** 2
        ss.append(s)

        if s > s_max[1]:
            s_max = (threshold, s)
            
    return s_max[0]

# try to ge the outlines of the immage

gray_im = to_grayscale(im)

t = otsu_threshold(gray_im)
plti(simple_threshold(gray_im, t), cmap='Greys')

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

c_ims = []
for c, ax in zip(range(3), axs):
    tmp_im = im[:,:,c]
    t = otsu_threshold(tmp_im)
    tmp_im = simple_threshold(tmp_im, t)
    ax.imshow(tmp_im, cmap='Greys')
    c_ims.append(tmp_im)
    ax.set_axis_off()
   
    
   
from scipy.ndimage.interpolation import zoom
im_small = zoom(im, (0.2,0.2,1))

plti(im_small)
   
    
 # use Kmeans to find color clusters  
    
    
plti(c_ims[0] & c_ims[1] & c_ims[2], cmap='Greys')

from sklearn.cluster import KMeans

h,w = im_small.shape[:2]
im_small_long = im_small.reshape((h * w, 3))
im_small_wide = im_small_long.reshape((h,w,3))

km = KMeans(n_clusters=3)

km.fit(im_small_long)

cc = km.cluster_centers_.astype(np.uint8)
out = np.asarray([cc[i] for i in km.labels_]).reshape((h,w,3))

plti(out)

dog = np.asarray([cc[i] if i == 1 else [0,0,0]
                  for i in km.labels_]).reshape((h,w,3))

plti(dog)


dog


# lets try this on another image


im = plt.imread(r"C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\image_16.jpg")


# pixel threshholds were pixel are seens as the same
thresholds = [100,120,128,138,150]


fig, axs = plt.subplots(nrows=1, ncols=len(thresholds), figsize=(20,5));

# make gray image
gray_im = to_grayscale(im)

plti(gray_im, cmap='Greys')


t = otsu_threshold(gray_im)
plti(simple_threshold(gray_im, t), cmap='Greys')

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

c_ims = []
for c, ax in zip(range(3), axs):
    tmp_im = im[:,:,c]
    t = otsu_threshold(tmp_im)
    tmp_im = simple_threshold(tmp_im, t)
    ax.imshow(tmp_im, cmap='Greys')
    c_ims.append(tmp_im)
    ax.set_axis_off()
   
# create smaller picture size   
   
im_small = zoom(im, (0.2,0.2,1))

plti(im_small)
   
    
 # use Kmeans to find color clusters  
    
    
plti(c_ims[0] & c_ims[1] & c_ims[2], cmap='Greys')


h,w = im_small.shape[:2]
im_small_long = im_small.reshape((h * w, 3))
im_small_wide = im_small_long.reshape((h,w,3))

km = KMeans(n_clusters=3)

km.fit(im_small_long)

cc = km.cluster_centers_.astype(np.uint8)
out = np.asarray([cc[i] for i in km.labels_]).reshape((h,w,3))

plti(out)

dog = np.asarray([cc[i] if i == 1 else [0,0,0]
                  for i in km.labels_]).reshape((h,w,3))

plti(dog)

# another weapon try

im = plt.imread(r"C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\image_8.jpg")


# pixel threshholds were pixel are seens as the same
thresholds = [100,120,128,138,150]


fig, axs = plt.subplots(nrows=1, ncols=len(thresholds), figsize=(20,5));

# make gray image
gray_im = to_grayscale(im)

plti(gray_im, cmap='Greys')


t = otsu_threshold(gray_im)
plti(simple_threshold(gray_im, t), cmap='Greys')

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

c_ims = []
for c, ax in zip(range(3), axs):
    tmp_im = im[:,:,c]
    t = otsu_threshold(tmp_im)
    tmp_im = simple_threshold(tmp_im, t)
    ax.imshow(tmp_im, cmap='Greys')
    c_ims.append(tmp_im)
    ax.set_axis_off()
   
# create smaller picture size   
   
im_small = zoom(im, (0.2,0.2,1))

plti(im_small)
   
    
 # use Kmeans to find color clusters  
    
    
plti(c_ims[0] & c_ims[1] & c_ims[2], cmap='Greys')


h,w = im_small.shape[:2]
im_small_long = im_small.reshape((h * w, 3))
im_small_wide = im_small_long.reshape((h,w,3))

km = KMeans(n_clusters=3)

km.fit(im_small_long)

cc = km.cluster_centers_.astype(np.uint8)
out = np.asarray([cc[i] for i in km.labels_]).reshape((h,w,3))

plti(out)

dog = np.asarray([cc[i] if i == 1 else [0,0,0]
                  for i in km.labels_]).reshape((h,w,3))

plti(dog)

# indeed very similar to other wepon



# lets try converting the pictures this way and see if the model can identify the more easily

list_all_arrays = []
for x in range(1,48):
    linkstring=r"C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\image_" + str(x)+ ".jpg"
    #print(linkstring)
    
    # resize
    
    pic = np.array(Image.open(linkstring).resize((150,150)))
    #im_small = im_small = zoom(pic, (0.2,0.2,1))
    # kmeans
    h,w = pic.shape[:2]
    im_small_long = pic.reshape((h * w, 3))
    im_small_wide = im_small_long.reshape((h,w,3))

    km = KMeans(n_clusters=2)

    km.fit(im_small_long)

    cc = km.cluster_centers_.astype(np.uint8)
    out = np.asarray([cc[i] for i in km.labels_]).reshape((h,w,3))


    dog = np.asarray([cc[i] if i == 1 else [0,0,0]
                      for i in km.labels_]).reshape((h,w,3))

    #plti(dog)
    
    # flatten
    array2 = dog.flatten()

    #Memory occupied by array 2

    #displaying the 1-D array
    #print(array2)
    list_all_arrays.append(array2)


lists_in_list = [list(x) for x in list_all_arrays]
lists_in_list
finished_array = np.array(lists_in_list)
finished_array


finarrayfeatures1 = np.apply_along_axis(standardize,axis=0, arr=finished_array)
finarrayfeatures1


# might be two small the image decomposition but lets try

X_train, X_test, y_train, y_test = train_test_split(finarrayfeatures1, Y, test_size=0.20, random_state=14)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(90, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(70, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(68, activation=tf.nn.softmax))

tf.keras.optimizers.Adam(
    learning_rate=0.3,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)


model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(X_train,y_train, epochs=120)



val_loss, val_acc = model.evaluate(X_test, y_test)


# very decent result with different test samples
model.save('pixel_work1.model')

testmodel1 = tf.keras.models.load_model('pixel_work1.model')

val_loss, val_acc = testmodel1.evaluate(X_test, y_test)
'''
val_loss, val_acc = testmodel1.evaluate(X_test, y_test)
1/1 [==============================] - 0s 283ms/step - loss: 0.9207 - accuracy: 0.8667

X_train, X_test, y_train, y_test = train_test_split(finarrayfeatures1, Y, test_size=0.30, random_state=20)

val_loss, val_acc = testmodel1.evaluate(X_test, y_test)
1/1 [==============================] - 0s 50ms/step - loss: 2.0250 - accuracy: 0.8000

X_train, X_test, y_train, y_test = train_test_split(finarrayfeatures1, Y, test_size=0.40, random_state=27)

val_loss, val_acc = testmodel1.evaluate(X_test, y_test)
1/1 [==============================] - 0s 33ms/step - loss: 1.6282 - accuracy: 0.8421

X_train, X_test, y_train, y_test = train_test_split(finarrayfeatures1, Y, test_size=0.20, random_state=42)

val_loss, val_acc = testmodel1.evaluate(X_test, y_test)
1/1 [==============================] - 0s 100ms/step - loss: 3.3038 - accuracy: 0.6000

X_train, X_test, y_train, y_test = train_test_split(finarrayfeatures1, Y, test_size=0.20, random_state=14)

val_loss, val_acc = testmodel1.evaluate(X_test, y_test)
1/1 [==============================] - 0s 41ms/step - loss: 1.5381 - accuracy: 0.8000


'''

# lets try it with even bigger resize

list_all_arrays = []
for x in range(1,48):
    linkstring=r"C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\image_" + str(x)+ ".jpg"
    #print(linkstring)
    
    # resize
    
    pic = np.array(Image.open(linkstring).resize((250,250)))
    #im_small = im_small = zoom(pic, (0.2,0.2,1))
    # kmeans
    h,w = pic.shape[:2]
    im_small_long = pic.reshape((h * w, 3))
    im_small_wide = im_small_long.reshape((h,w,3))

    km = KMeans(n_clusters=2)

    km.fit(im_small_long)

    cc = km.cluster_centers_.astype(np.uint8)
    out = np.asarray([cc[i] for i in km.labels_]).reshape((h,w,3))


    dog = np.asarray([cc[i] if i == 1 else [0,0,0]
                      for i in km.labels_]).reshape((h,w,3))

    plti(dog)
    
    # flatten
    array2 = dog.flatten()

    #Memory occupied by array 2

    #displaying the 1-D array
    #print(array2)
    list_all_arrays.append(array2)


lists_in_list = [list(x) for x in list_all_arrays]
lists_in_list
finished_array = np.array(lists_in_list)
finished_array


finarrayfeatures3 = np.apply_along_axis(standardize,axis=0, arr=finished_array)
finarrayfeatures3


# might be two small the image decomposition but lets try

X_train, X_test, y_train, y_test = train_test_split(finarrayfeatures3, Y, test_size=0.80, random_state=46)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(90, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(70, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(68, activation=tf.nn.softmax))

tf.keras.optimizers.Adam(
    learning_rate=0.3,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)


model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(X_train,y_train, epochs=120)



val_loss, val_acc = model.evaluate(X_test, y_test)

# very very decent results as well

model.save('pixel_work2.model')


# lets try with even more epochs

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(90, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(70, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(68, activation=tf.nn.softmax))

tf.keras.optimizers.Adam(
    learning_rate=0.3,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)


model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(X_train,y_train, epochs=300)



val_loss, val_acc = model.evaluate(X_test, y_test)

model.save('pixel_change3.model')
testmodel3 = tf.keras.models.load_model('pixel_change3.model')

val_loss, val_acc = testmodel3.evaluate(X_test, y_test)
'''

val_loss, val_acc = testmodel3.evaluate(X_test, y_test)
2/2 [==============================] - 0s 20ms/step - loss: 1.9987 - accuracy: 0.7368

X_train, X_test, y_train, y_test = train_test_split(finarrayfeatures3, Y, test_size=0.80, random_state=32)

val_loss, val_acc = testmodel3.evaluate(X_test, y_test)
2/2 [==============================] - 0s 16ms/step - loss: 2.0016 - accuracy: 0.7368

X_train, X_test, y_train, y_test = train_test_split(finarrayfeatures3, Y, test_size=0.80, random_state=46)

val_loss, val_acc = testmodel3.evaluate(X_test, y_test)
2/2 [==============================] - 0s 23ms/step - loss: 1.4372 - accuracy: 0.7895

'''



# test station
dog.flatten()

pic = np.array(Image.open(r"C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\image_8.jpg").resize((80,80)))

pic.shape


# lets try other rearranging pixel methods

# the out result



list_all_arrays = []
for x in range(1,48):
    linkstring=r"C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\image_" + str(x)+ ".jpg"
    #print(linkstring)
    
    # resize
    
    pic = np.array(Image.open(linkstring).resize((250,250)))
    #im_small = im_small = zoom(pic, (0.2,0.2,1))
    # kmeans
    h,w = pic.shape[:2]
    im_small_long = pic.reshape((h * w, 3))
    im_small_wide = im_small_long.reshape((h,w,3))

    km = KMeans(n_clusters=2)

    km.fit(im_small_long)

    cc = km.cluster_centers_.astype(np.uint8)
    out = np.asarray([cc[i] for i in km.labels_]).reshape((h,w,3))


    #dog = np.asarray([cc[i] if i == 1 else [0,0,0]
          #            for i in km.labels_]).reshape((h,w,3))

    plti(out)
    
    # flatten
    array2 = out.flatten()

    #Memory occupied by array 2

    #displaying the 1-D array
    #print(array2)
    list_all_arrays.append(array2)


lists_in_list = [list(x) for x in list_all_arrays]
lists_in_list
finished_array = np.array(lists_in_list)
finished_array


finarrayfeatures = np.apply_along_axis(standardize,axis=0, arr=finished_array)
finarrayfeatures

# apply new data arrays
X_train, X_test, y_train, y_test = train_test_split(finarrayfeatures, Y, test_size=0.20, random_state=54)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(90, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(70, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(68, activation=tf.nn.softmax))

tf.keras.optimizers.Adam(
    learning_rate=0.3,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)


model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(X_train,y_train, epochs=200)
'''
1/1 [==============================] - 0s 319ms/step - loss: 0.2022 - accuracy: 0.9000

X_train, X_test, y_train, y_test = train_test_split(finarrayfeatures, Y, test_size=0.20, random_state=12)

val_loss, val_acc = model.evaluate(X_test, y_test)
1/1 [==============================] - 0s 40ms/step - loss: 0.2408 - accuracy: 0.9000

X_train, X_test, y_train, y_test = train_test_split(finarrayfeatures, Y, test_size=0.20, random_state=54)

val_loss, val_acc = model.evaluate(X_test, y_test)
1/1 [==============================] - 0s 38ms/step - loss: 0.0138 - accuracy: 1.0000


'''


val_loss, val_acc = model.evaluate(X_test, y_test)


model.save('pixel_change4.model')


# lets try all grey

list_all_arrays = []
for x in range(1,48):
    linkstring=r"C:\Users\leube\Ironhack\Ironprojects\Module_3\final_project\the-dark-side-of-social-media-\images\image_" + str(x)+ ".jpg"
    #print(linkstring)
    
    # resize
    
    pic = np.array(Image.open(linkstring).resize((250,250)))
    
    im = to_grayscale(pic)

    # flatten
    array2 = im.flatten()

    list_all_arrays.append(array2)


lists_in_list = [list(x) for x in list_all_arrays]
lists_in_list
finished_array = np.array(lists_in_list)
finished_array


finarrayfeatures5 = np.apply_along_axis(standardize,axis=0, arr=finished_array)
finarrayfeatures5

# apply new data arrays
X_train, X_test, y_train, y_test = train_test_split(finarrayfeatures5, Y, test_size=0.40, random_state=25)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(90, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(70, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(68, activation=tf.nn.relu))

tf.keras.optimizers.Adam(
    learning_rate=0.3,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)


model.compile(optimizer='Adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(X_train,y_train, epochs=160)


val_loss, val_acc = model.evaluate(X_test, y_test)
'''
1/1 [==============================] - 0s 37ms/step - loss: 0.5693 - accuracy: 0.8000

X_train, X_test, y_train, y_test = train_test_split(finarrayfeatures5, Y, test_size=0.30, random_state=29)

val_loss, val_acc = model.evaluate(X_test, y_test)
1/1 [==============================] - 0s 55ms/step - loss: 0.4818 - accuracy: 0.8667

X_train, X_test, y_train, y_test = train_test_split(finarrayfeatures5, Y, test_size=0.40, random_state=21)

val_loss, val_acc = model.evaluate(X_test, y_test)
1/1 [==============================] - 0s 53ms/step - loss: 0.4547 - accuracy: 0.8421

X_train, X_test, y_train, y_test = train_test_split(finarrayfeatures5, Y, test_size=0.40, random_state=25)

val_loss, val_acc = model.evaluate(X_test, y_test)
1/1 [==============================] - 0s 40ms/step - loss: 0.2152 - accuracy: 0.8421


'''

# lets increase the size of my data



