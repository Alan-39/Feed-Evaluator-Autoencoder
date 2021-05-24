import os
import numpy as np
import imageio
import PIL
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

"""
Functions for loading and preprocessing images
"""

# loads single image numpy array
def load_img_arr(img_path, img_size):
    #img = cv2.imread(img_path)
    #img = cv2.resize(img, img_size)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    currimg = imageio.imread(img_path) 
    im = np.array(PIL.Image.fromarray(currimg).resize((img_size[0], img_size[1])))
    im = im[:, :, :3]
    #im = np.array(load_img(img_path, target_size=img_size))
    return im


# loads single gt numpy array
def load_gt_arr(gt_path, img_size):
    currimg = imageio.imread(gt_path)
    im = np.array(PIL.Image.fromarray(currimg).resize((img_size[0], img_size[1])))
    #im = im[:, :, :1]
    return im

# Deprecated, appending individual loaded img numpy array into array is amazingly inefficient and slow
# especially for large sample size, use Dataset class implementation instead
def load_dataset(img_path, gt_path, img_size):
    imgcnt = 0
    image_arr = []
    label_arr = []

    name = sorted(os.listdir(img_path))
    for text in name:
        fullpath = img_path + "\\" + text
        image_arr.append(load_img_arr(fullpath, img_size))
        imgcnt = imgcnt + 1
    name = sorted(os.listdir(gt_path))
    for text in name:
        fullpath = gt_path + "\\" + text
        label_arr.append(load_gt_arr(fullpath, img_size))
    print("Total images: %d" % (imgcnt))

    return image_arr, label_arr, imgcnt


def load_img_paths(img_path):
    img_paths = sorted(
        [
            os.path.join(img_path, fname)
            for fname in os.listdir(img_path)
            if fname.endswith(".jpg") or fname.endswith(".png")
        ]
    )
    return img_paths


def load_gt_paths(gt_path):
    gt_paths = sorted(
        [
            os.path.join(gt_path, fname)
            for fname in os.listdir(gt_path)
            if fname.endswith(".png")
        ]
    )
    return gt_paths

class Dataset(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = img
        return x, y