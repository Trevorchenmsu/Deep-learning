import os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
import h5py


def imshow(img):
    cv2.imshow("img", img)
    cv2.waitKey(50000)


def get_multi_box_attr(mat, box_attr):
    len_attr = len(box_attr)
    result = [0] * len_attr

    if len(box_attr) > 1:
        for i in range(len_attr):
            ref = box_attr[i, 0]
            ref_np = np.array(mat[ref])
            result[i] = int(ref_np[0, 0])
    else:
        result[0] = int(box_attr[0, 0])

    return result


def get_img_name(name_ref):
    name_ref = np.array(name_ref)

    name = ''
    for i in range(len(name_ref)):
        name += (chr(name_ref[i, 0]))

    return name


def load_images_from_dir(data_dir, ext=".png"):
    imagesFiles = [f for f in os.listdir(data_dir) if f.endswith(ext)]
    imgs = [np.array(cv2.imread(os.path.join(data_dir, f), 0)) for f in imagesFiles]
    return imgs


def load_images_from_filename(data_dir, filenames):
    imagesFiles = [os.path.join(data_dir, filename) for filename in filenames]
    imgs = [np.array(cv2.imread(os.path.join(data_dir, f), 0)) for f in imagesFiles]
    return imgs

# fix the length of each label as 6
def get_final_labels(labels):
    n_label = len(labels)
    final_labels = [[11 for j in range(6)] for i in range(n_label)]
    for i, label in enumerate(labels):
        if not isinstance(label, list):
            final_labels[i][0] = label
        else:
            for j in range(len(label)):
                final_labels[i][j] = label[j]
            final_labels[i][5] = len(label)

    final_labels = np.array(final_labels)
    final_labels[final_labels == 10] = 0
    final_labels[final_labels == 11] = 10

    return final_labels


# load .mat format data
def load_mat_data(mat_dir):
    mat = h5py.File(mat_dir, 'r')  # key: ['#refs#', 'digitStruct']
    digits_struct = mat['digitStruct']  # key: ['bbox', 'name']
    bbox_ref = digits_struct['bbox']
    name_ref = digits_struct['name']

    n_bbox = len(bbox_ref)
    bbox = [{}] * n_bbox
    # print(mat[bbox_ref[5].item()].keys()) # keys: ['height', 'label', 'left', 'top', 'width']
    filenames = [''] * n_bbox
    labels = [0] * n_bbox

    for i in range(n_bbox):
        bbox_keys = mat[bbox_ref[i].item()]

        height = bbox_keys['height']
        label = bbox_keys['label']
        left = bbox_keys['left']
        top = bbox_keys['top']
        width = bbox_keys['width']

        ht = get_multi_box_attr(mat, height)
        lb = get_multi_box_attr(mat, label)
        lf = get_multi_box_attr(mat, left)
        tp = get_multi_box_attr(mat, top)
        wd = get_multi_box_attr(mat, width)

        filenames[i] = get_img_name(mat[name_ref[i, 0]])
        labels[i] = lb

        bb = {'left': lf, 'top': tp, 'width': wd, 'height': ht}

        bbox[i] = bb

    return bbox, filenames, labels

def ground_truth_bbox(bbox):
    gt_bbox = [0] * len(bbox)
    for i, box in enumerate(bbox):
        left = box['left']
        top = box['top']
        height = box['height']
        width = box['width']

        x = int(min(left))
        y = int(min(top))
        w = int(left[-1] + width[-1])
        h = int(max(top) + max(height))

        gt_bbox[i] = (x, y, w, h)

    return gt_bbox

def draw_box(img, box):
    left = box['left']
    top = box['top']
    height = box['height']
    width = box['width']

    x = int(min(left))
    y = int(min(top))
    w = int(left[-1] + width[-1])
    h = int(max(top) + max(height))

    return cv2.rectangle(img.copy(), (x, y), (w, h), (0, 255, 0), 2)


if __name__ == '__main__':
    # train_mat = "./train/digitStruct.mat"
    # test_mat = "./test/digitStruct.mat"
    # bbox, filenames, labels = load_mat_data(train_mat)
    # final_labels = get_final_labels(labels)

    # display
    # plt.figure(figsize=(20, 8))
    # for idx, i in enumerate(range(12500, 12515)):
    #     img_dir = os.path.join('./train/', filenames[i])
    #     img = cv2.imread(img_dir)
    #     box = bbox[i]
    #
    #     bb = draw_box(img, box)
    #
    #     plt.subplot(3, 5, idx + 1)
    #     plt.title(final_labels[i])
    #     plt.imshow(cv2.cvtColor(bb, cv2.COLOR_BGR2RGB))
    #     plt.axis("off")
    #
    # plt.show()

    # get training dataset
    train_mat = "./train/digitStruct.mat"
    bbox_train, filenames_train, labels_train = load_mat_data(train_mat)
    Xtrain = load_images_from_filename(filenames_train) # gray scale
    # train_dir = "./train/"
    # Xtrain = load_images_from_dir(train_dir)
    ytrain = get_final_labels(labels_train)
    bbox_gt_train = ground_truth_bbox(bbox_train)
    print("The shape of training dataset is: ", len(Xtrain))
    print("***********************************", len(Xtrain))

    # get testing dataset
    test_mat = "./test/digitStruct.mat"

    bbox_test, filenames_test, labels_test = load_mat_data(train_mat)
    Xtest = load_images_from_filename(filenames_test) # gray scale
    # test_dir = "./test/"
    # Xtest = load_images_from_dir(test_dir)
    ytest = get_final_labels(labels_test)
    bbox_gt_test = ground_truth_bbox(bbox_test)

    print("The shape of testing dataset is: ", len(Xtest))
    print("***********************************", len(Xtrain))








