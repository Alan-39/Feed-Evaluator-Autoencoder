import cv2
import base64
import json
import os
import imutils
from labelme import utils
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt


"""
Assorted functions for conversion, contour finding, etc
"""

# takes contour from predicted output and draws it on original image
def draw_contour(img_pred, img_original):
    cnt = cv2.findContours(img_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contour = cv2.drawContours(img_original, cnt[0], -1, (64, 224, 208), 3)
    return img_contour


# takes contour from predicted output and fills it on blank image
def draw_filled_contour(img_pred, img_size):
    cnt = cv2.findContours(img_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blank_image = np.zeros((img_size[0], img_size[1], 3), np.uint8)
    filled_contour = cv2.drawContours(blank_image, cnt[0], -1, (255, 255, 255), -1)
    return filled_contour


def segment_background(img_path, mask_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (512, 512))
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    out = np.zeros_like(image) # Extract out the object and place into output image
    out[mask > 50] = image[mask > 50]

    (y, x) = np.where(mask == 0)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = out[topy:bottomy+1, topx:bottomx+1]

    # Show the output image
    #cv2.imshow('original', image)
    #cv2.imshow('mask', mask)
    #cv2.imshow('Output', out)

    k = 3
    data = np.float32(out).reshape(-1,3)

    ret, labels, colours = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 64, 1), 10, cv2.KMEANS_RANDOM_CENTERS)

    colours = np.uint8(colours)
    labels = colours[labels.flatten()]
    labels = labels.reshape(out.shape)

    gray = cv2.cvtColor(labels, cv2.COLOR_BGR2GRAY)
    threshold = img_threshold(gray)

    cv2.imshow('gray', gray)
    cv2.imshow('threshold', threshold)
    cv2.imshow('k means cluster', labels)
    cv2.waitKey(0)


def calc_whitepixels(filled_contour):
    height, width = filled_contour.shape[0], filled_contour.shape[1]
    img_mean = cv2.mean(filled_contour)[0]/255

    pixels = height * width
    #print("Total image pixels:", pixels)

    area = img_mean * pixels
    #print("white pixel filled:", area)

    percentage = area / pixels * 100
    print("percentage filled:", percentage)
    return percentage


def img_threshold(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.threshold(blurred, 12, 255, cv2.THRESH_BINARY)[1]
    return thresh


# converts coco json dataset into img and gt mask
def labelmejson_to_dataset(json_path):
    json_paths = sorted(
        [
            os.path.join(json_path, fname)
            for fname in os.listdir(json_path)
            if fname.endswith(".json")
        ]
    )

    os.makedirs("converted_dataset\\img", exist_ok=True)
    os.makedirs("converted_dataset\\gt", exist_ok=True)
    for i in range(len(json_paths)):
        data = json.load(open(json_paths[i]))
        imageData = data.get("imageData")

        # ignore "json_file" is not defined warning
        if not imageData:
            imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
            with open(imagePath, "rb") as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode("utf-8")
        img = utils.img_b64_to_arr(imageData)

        label_name_to_value = {"_background_": 0}
        for shape in sorted(data["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
        lbl, _ = utils.shapes_to_label(img.shape, data["shapes"], label_name_to_value)

        img = img[:, :, :3]
        PIL.Image.fromarray(img).save(os.path.join("converted_dataset\\img", "{}.jpg".format(i)))
        utils.lblsave(os.path.join("converted_dataset\\gt", "{}.png".format(i)), lbl)


# converts gt mask into gt dots by finding contour of mask, calculate its centroid and drawing a dot
def gtmask_to_gtdots(gt_path):
    gt_paths = sorted(
        [
            os.path.join(gt_path, fname)
            for fname in os.listdir(gt_path)
            if fname.endswith(".png")
        ]
    )
    os.makedirs("gt-dots", exist_ok=True)
    for i in range(len(gt_paths)):
        image = cv2.imread(gt_paths[i])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = img_threshold(gray)

        height, width, channels = image.shape
        blank_image = np.zeros((height, width, channels), np.uint8)

        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.circle(blank_image, (cX, cY), 2, (255, 255, 255), -1)
        cv2.imwrite("gt-dots\\{}.jpg".format(i), blank_image)


# shows model training history
def show_training_history(history_data):
    plt.figure(1)  
    
    # show training history for accuracy  
    plt.subplot(211)  
    plt.plot(history_data['accuracy'])  
    plt.plot(history_data['val_accuracy'])  
    plt.title('Model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'val'], loc='upper left')  
    
    # show training history for loss  
    plt.subplot(212)  
    plt.plot(history_data['loss'])  
    plt.plot(history_data['val_loss'])  
    plt.title('Model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'val'], loc='upper left')  
    plt.show()
