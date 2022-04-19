import os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
import h5py


import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



#!~/anaconda3/envs/CS6476/bin/python

# pytorch related packages
import cnn
import torch

# additional packages
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import common

# define constants
SATURATION = 255    # intensity saturation value
COLOR_RED = (0, 0, SATURATION)  # color red in BGR space
TEST_DATA_DIR = "./test_image"  # directory of the images for test
ASPECT_RATIO_UPPER = 0.8    # upper limit of the aspect ratio of the bounding box that may contain a number
ASPECT_RATIO_LOWER = 0.3    # lower limit of the aspect ratio of the bounding box that may contain a number
MARGIN_MAX_MEAN = 1.3   # margin between the largest value and mean value of the predicted result
MARGIN_MAX_MAX = 1.3    # margin between the largest value and mean value of the predicted result


class TestingDataPackage:
    """
    This class is used to package and transport the data to test the CNN model
    The original image and a list of bounding box are needed to instantiate the class
    The final image and bbox_house_number are yielded by applied the CNN model on the original image
    """
    def __init__(self, img, bbox_list):
        self.original_img = img             # original image
        self.original_bbox_list = bbox_list # original list of bounding box
        self.final_img = None               # bounding box + house number overlaid on the original image
        self.final_bbox_list = None         # final list of bounding box and house number


def get_bounding_box(img_in, show_img=False):
    """
    :param img_in: input BGR image
    :param show_img: bool value to determine if to show the image with bounding box
    :return:
    - bbox_list: list of bounding box defined by x, y, w, h where (x, y) is the coordinate of hte upper left corner and
      (w, h) is the width and height of the bounding box
    """

    # deep copy of the original BGR image, then convert it into grayscale
    vis = np.copy(img_in)
    img_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

    """
    - instantiate the MSER object, and apply it on the image, which returns the bounding boxes that potentially contain 
      a number
    - regroup the bounding boxes
    - traverse the boundary boxes, and use aspect ratio of the box to screen the box that unlikely contains a number. 
      the upper and lower limit of the bounding box that likely contains a number is defined as a global variable 
    """
    mser = cv2.MSER_create()
    _, bboxes = mser.detectRegions(img_gray)
    bboxes, _ = cv2.groupRectangles(bboxes.tolist(), groupThreshold=8, eps=0.2)
    bbox_list = []
    for bbox in bboxes:
        x, y, w, h = bbox
        cond1 = ASPECT_RATIO_LOWER <= w / h <= ASPECT_RATIO_UPPER
        cond2 = w > 10 and h > 20
        if cond1 and cond2:
            cv2.rectangle(vis, (x, y), (x + w, y + h), COLOR_RED, 1)
            bbox_list.append((x, y, w, h))

    # for debug purpose to show the image
    if show_img:
        cv2.imshow(" ", vis)
        cv2.waitKey(0)

    return bbox_list


def apply_cnn_model(model, data_package, criterion=None, show_patch=False):
    """
    :param:
        model: a pretrained CNN model
        data_package: a data structure to transport data
    :return:
        none
    """
    selected_bboxes = []
    for box in data_package.original_bbox_list:
        """
        Notifications
        1. testing images are initially in cv2 format, i.e. BGR and [0, 255]
        2. patches extracted from cv2 images, and resized it to 32 x 32
        3. in order to feed the patches into the CNN model, cv2 images needs to be converted to PIL format, 
           i.e. RBG [0.0, 1.0]
        4. apply the model on the patch, and predict the number in it
            4.1 extract the index of the largest score
            4.2 the difference between the largest score and the average should be largeer than a predefined margin
        """
        x, y, w, h = box
        img = cv2.GaussianBlur(data_package.original_img, (3, 3), 0)
        patch = img[y:y + h, x:x + w, ::]
        patch = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_AREA)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch = Image.fromarray(patch)

        # for debug purpose, plot out the patch
        if show_patch:
            plt.imshow(patch)
            plt.show()

        patch = common.TRANSFORM(patch)
        outputs = model(patch[None, ...])
        predicted = outputs.data.numpy()
        predicted = predicted[0]

        if criterion == "MAX_MEAN":
            mean_val, max_val = np.mean(predicted), np.amax(predicted)
            if max_val >= mean_val + MARGIN_MAX_MEAN:
                idx_max_val = np.where(predicted == max_val)[0][0]
                selected_bboxes.append((box, idx_max_val))

        if criterion == "MAX_MAX":
            ordered_by_score = np.sort(predicted)
            max_val_1 = ordered_by_score[9]
            max_val_2 = ordered_by_score[8]
            if max_val_1 > max_val_2 + MARGIN_MAX_MAX:
                idx_max_val = np.where(predicted == max_val_1)[0][0]
                selected_bboxes.append((box, idx_max_val))

        if not len(selected_bboxes) == 0:
            data_package.final_bbox_list = selected_bboxes
        else:
            print("we do not find any number in this image")


def get_final_result(img_name, data_package, mode=None):

    img = data_package.original_img

    for bbox in data_package.final_bbox_list:
        (x, y, w, h), val = bbox
        img = cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_RED, 1)
        img = cv2.putText(img, '{}'.format(val), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, COLOR_RED, 2, cv2.LINE_AA)

    if mode == "show":
        # show the marked images
        cv2.imshow(" ", img)
        cv2.waitKey(0)

    if mode == "write":
        # save the marked images
        img_name = img_name.split('_')[-1]
        img_name = img_name.split('.')[0]
        cv2.imwrite("./{}.png".format(img_name), img)


def main():

    os.system("clear")
    img_list = glob.glob("{}/*.jpg".format(TEST_DATA_DIR))
    img_list.sort()
    # load the original image, and get bounding boxes
    cnn_test_dict = {}
    for img_name in img_list:
        img = cv2.imread(img_name)
        bbox_list = get_bounding_box(img, show_img=False)
        cnn_test_dict.update({img_name: TestingDataPackage(img, bbox_list)})

    # load the pretrained CNN model
    print("Load the model")
    model = cnn.CNN()
    try:
        model.load_state_dict(torch.load(common.PATH_CLASSIFIER))
        print("\n\nThe model is found under {}".format(common.PATH_CLASSIFIER))
    except ValueError:
        print("\n\nThe model cannot be found under {}".format(common.PATH_CLASSIFIER))
        raise NotImplementedError

    # apply CNN model on the images
    print("\n\nStarting traversing the test data set")
    for key, data_package in cnn_test_dict.items():
        print("\tprocessing {}".format(key))
        apply_cnn_model(model, data_package, criterion="MAX_MEAN", show_patch=False)
        get_final_result(key, data_package, mode="write")


if __name__ == "__main__":
    main()