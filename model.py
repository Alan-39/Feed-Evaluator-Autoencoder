from preprocessor import *
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
import matplotlib.pyplot as plt
import random
import json
import numpy as np
import math
from utilities import *

"""
Functions for model architecture, training and prediction.
"""

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("No GPU devices detected. Running on CPU")

img_size = (512, 512)
epochs = 5
batch_size = 4


def unet_model():
    """U-Net architecture based on paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox"""
    inputs = Input(shape=(img_size[0], img_size[1], 3))

    conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool1)
    conv2 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), padding='same', activation='relu')(pool2)
    conv3 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool3)
    conv4 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), padding='same', activation='relu')(pool4)
    conv5 = Conv2D(1024, (3, 3), padding='same', activation='relu')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), padding='same', activation='relu')(up6)
    conv6 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), padding='same', activation='relu')(up7)
    conv7 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), padding='same', activation='relu')(up8)
    conv8 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), padding='same', activation='relu')(up9)
    conv9 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    #model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def autoencoder_model():
    """simple autoencoder model"""
    inputs = Input(shape=(img_size[0], img_size[1], 3))

    conv1 = Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (5, 5), activation='relu', padding='same')(pool1)

    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2) 
    up1 = UpSampling2D((2, 2))(conv3)
    decoded = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(up1)

    model = Model(inputs=inputs, outputs=decoded)
    #model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_model(img_path, gt_path, model_name):
    """loading dataset using load_dataset() is now deprecated, leftover code just in case as a fall back measure."""

#    x_train, y_train, imgcnt = load_dataset(img_path, gt_path, img_size)
#    x_train = np.asarray(x_train).reshape(imgcnt, img_size[0], img_size[1], 3)
#    y_train = np.asarray(y_train).reshape(imgcnt, img_size[0], img_size[1], 1)

#    model = unet_model(img_size)
#    callbacks = [keras.callbacks.ModelCheckpoint("models/{}.h5".format(model_name), save_best_only=True)]
#    history = model.fit(x=x_train, y=y_train, validation_split=0.1, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=1)

    input_img_paths = load_img_paths(img_path)
    target_img_paths = load_gt_paths(gt_path)

    # randomly shuffle lists and splits training and validation dataset into 90/10
    val_samples = math.ceil(len(input_img_paths) * 0.1)
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    train_gen = Dataset(batch_size, img_size, train_input_img_paths, train_target_img_paths)
    val_gen = Dataset(batch_size, img_size, val_input_img_paths, val_target_img_paths)

    model = autoencoder_model()
    callbacks = [keras.callbacks.ModelCheckpoint("models/{}.h5".format(model_name), save_best_only=True)]
    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks, verbose=1)

    # Saves model training history as json
    os.makedirs("model_training_history", exist_ok=True)
    with open('model_training_history/{}_history.json'.format(model_name), 'w') as f:
        json.dump(history.history, f)    
    show_training_history(history.history)


def predict_image(model_path, img_path):
    import time
    start = time.time()

    model = load_model(model_path)
    model.summary()
    
    img = load_img_arr(img_path, img_size)

    fig, axs = plt.subplots(1, 5, figsize=(15, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title('Original image')

    X_test = np.reshape(img, (-1, img_size[0], img_size[1], 3))
    preds = model.predict(X_test)
    pred = np.reshape(preds, (img_size[0], img_size[1]))

    pred_uint8 = pred * 255
    pred_uint8 = pred_uint8.astype(np.uint8)
    bw = img_threshold(pred_uint8)

    contour = draw_contour(bw, img)

    plt.subplot(1, 4, 4)
    plt.imshow(pred)
    plt.title('Predicted output')

    plt.subplot(1, 4, 3)
    plt.imshow(bw, cmap='gray')
    plt.title('BW')

    plt.subplot(1, 4, 2)
    plt.imshow(contour)
    plt.title('Predicted Contour')

    filled_contour = draw_filled_contour(bw, img_size)
    calc_whitepixels(filled_contour)

    end = time.time()
    print("runtime (in seconds): ", end - start)

    cv2.imwrite("mask.jpg", filled_contour)

    plt.savefig('plot_figure.png')
    plt.show()
    

def predict_batch(model_path, img_path):
    import time
    start = time.time()

    model = load_model(model_path)
    model.summary()

    img_paths = load_img_paths(img_path)
    print(img_paths)
    for i in range(len(img_paths)):
        img = load_img_arr(img_paths[i], img_size)

        X_test = np.reshape(img, (-1, img_size[0], img_size[1], 3))
        preds = model.predict(X_test)
        pred = np.reshape(preds, (img_size[0], img_size[1]))

        pred_uint8 = pred * 255
        pred_uint8 = pred_uint8.astype(np.uint8)
        bw = img_threshold(pred_uint8)

        contour = draw_contour(bw, img)

        plt.subplot(1, 4, 1)
        plt.imshow(img)
        plt.title('Original image')

        plt.subplot(1, 4, 4)
        plt.imshow(pred)
        plt.title('Predicted output')

        plt.subplot(1, 4, 3)
        plt.imshow(bw, cmap='gray')
        plt.title('BW')

        plt.subplot(1, 4, 2)
        plt.imshow(contour)
        plt.title('Predicted Contour')

        calc_whitepixels(bw, img_size)
        print("============")


    end = time.time()
    print("runtime (in seconds): ", end - start)

    print()
