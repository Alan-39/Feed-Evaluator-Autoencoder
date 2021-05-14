import os
import tensorflow as tf
import numpy as np
import skimage
from keras.models import Model, Sequential
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Input, Reshape
import imageio
from PIL import Image
from skimage import io, filters, measure
from scipy import ndimage
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# dataset source - http://visal.cs.cityu.edu.hk/downloads/smallobjects/

height, width = 410, 300
channel = 3 # 3 channels = RGB colours, 1 channel = grayscale
input_img = Input(shape = (height, width, channel))
batch_size = 5
epochs = 500

imgcnt = 0
imagevec = []
labelvec = []

path = "C:\\VSCode_Projects\\ObjCount_autoencoder\\dataset\\fish\\img"
name = sorted(os.listdir(path))
for text in name:
    fullpath = path + "\\" + text
    currimg = imageio.imread(fullpath) # returns array obtained by img
    im = np.array(Image.fromarray(currimg).resize((width, height))) # size = (width, height) as 2-tuple 
    imagevec.append(im)
    imgcnt = imgcnt + 1

path = "C:\\VSCode_Projects\\ObjCount_autoencoder\\dataset\\fish\\gt-dots"
name = sorted(os.listdir(path))
for text in name:
    fullpath = path + "\\" + text
    currimg = imageio.imread(fullpath)
    im = np.array(Image.fromarray(currimg).resize((width, height))) 
    labelvec.append(im)
print("Total images: %d" % (imgcnt))

datasize = imgcnt
batch_xs = np.asarray(imagevec)
batch_ys = np.asarray(labelvec)
x_train = batch_xs.reshape(datasize, height, width, channel)
y_train = batch_ys.reshape(datasize, height, width, 1)

def autoencoder(input_img):
    #encoder
    conv1 = Conv2D(16, (5, 5), activation='relu', padding='same')(input_img) 
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) 
    conv2 = Conv2D(32, (5, 5), activation='relu', padding='same')(pool1) 
    #decoder
    conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2) 
    up1= UpSampling2D((2, 2))(conv3)
    decoded = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(up1)
    return decoded

autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='binary_crossentropy', optimizer = 'adadelta')
autoencoder.summary()

autoencoder_train = autoencoder.fit(x_train, y_train, batch_size, epochs, verbose=1)

ind = np.random.randint(118)
fig, axs = plt.subplots(1, 5, figsize=(15, 4))
im = imagevec[ind]
label = labelvec[ind]

plt.subplot(1, 4, 1)
plt.imshow(im)
plt.title('Original image')

l = measure.label(im)
plt.subplot(1, 4, 2)
plt.imshow(label)
plt.title('Ground Truth image')

X_test = np.reshape(im, (-1, height, width, channel))
preds = autoencoder.predict(X_test)
pred = np.reshape(preds, (height, width))
plt.subplot(1, 4, 3)
plt.imshow(pred)
plt.title('Predicted image by model')

io.imsave('1.jpg', pred)
val = filters.threshold_sauvola(label)
drops = ndimage.binary_opening(label)
l = measure.label(drops)
pred = io.imread('1.jpg')
plt.subplot(1, 4, 4)
plt.title('BW predicted image')

val = filters.threshold_otsu(pred)
im2 = pred > val
plt.imshow(im2, cmap='gray')
drops = ndimage.binary_closing(pred)
l = measure.label(im2)
print("total number of fishes are: %d" %(l.max()))