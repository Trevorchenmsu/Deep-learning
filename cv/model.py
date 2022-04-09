# Define a Convolutional Neural Network
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.maxpool1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.maxpool2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.maxpool3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.maxpool4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.maxpool5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.dense = nn.Sequential(
            nn.Linear(512 * 1 * 1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )


    def forward(self, x):
        pool1 = self.maxpool1(x)
        pool2 = self.maxpool2(pool1)
        pool3 = self.maxpool3(pool2)
        pool4 = self.maxpool4(pool3)
        pool5 = self.maxpool5(pool4)

        flat = pool5.view(pool5.size(0), -1)
        class_ = self.dense(flat)
        return class_





# citation: https://cloud.tencent.com/developer/article/1760188
class CustomizedModel(nn.Module):
    def __init__(self):
        super(CustomizedModel, self).__init__()
        self.features = nn.Sequential(
            # layer 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False),
            nn.Dropout(p=0.2),

            # layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            # layer 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False),
            nn.Dropout(p=0.2),

            # layer 4
            nn.Conv2d(128, 180, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(180),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            # layer 5
            nn.Conv2d(180, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.2),

            # layer 6
            nn.Conv2d(256, 280, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(280),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            # layer 7
            nn.Conv2d(280, 300, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.2),

            # layer 8
            nn.Conv2d(300, 320, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1280, out_features=540, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=540, out_features=270, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=270, out_features=10, bias=True)
        )

    def forward(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out







#
#
# def flatten_images(batch_images):
#     """
#     Args:
#         batch_images: a batch of images
#     Returns:
#             X (numpy.array): data matrix of flatten images
#                              (row:observations, col:features) (float).
#     """
#
#     X = [0] * batch_images.shape[0]
#
#     for i, img in enumerate(batch_images):
#         img_flatten = img.flatten()
#         X[i] = img_flatten
#
#     X = np.array(X, dtype=np.float32)
#
#     return X
#
#
# def pca(X, k):
#     """PCA Reduction method.
#
#     Return the top k eigenvectors and eigenvalues using the covariance array
#     obtained from X.
#
#     Args:
#         X (numpy.array): 2D data array of flatten images (row:observations,
#                          col:features) (float).
#         k (int): new dimension space
#
#     Returns:
#         tuple: two-element tuple containing
#             eigenvectors (numpy.array): 2D array with the top k eigenvectors.
#             eigenvalues (numpy.array): array with the top k eigenvalues.
#     """
#     # 1. normalize the data to be zero mean
#     mu = np.mean(X, axis=0)
#     # face_centered = X - mean_face
#     cov_matrix = np.dot((X - np.array(mu, ndmin=2)).T, X - np.array(mu, ndmin=2))
#     eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
#     sort = eigenvalues.argsort()[::-1]
#     eigenvalues = eigenvalues[sort][:k]
#     eigenvectors = eigenvectors.T[sort][:k].T
#
#     return eigenvectors, eigenvalues
#
#
#
#
#
# class DesignModel(nn.Module):
#     def __init__(self):
#         super(DesignModel, self).__init__()
#         self.layer1 = nn.Sequential(
#                                     nn.Conv2d(3, 20, kernel_size=9, stride=1),
#                                     nn.BatchNorm2d(20),
#                                     nn.ReLU(),
#                                     nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False)
#                                    )
#
#         self.layer2 = nn.Sequential(
#                                     nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
#                                     nn.BatchNorm2d(20),
#                                     nn.ReLU(),
#
#                                     nn.Conv2d(20, 90, kernel_size=5, stride=1),
#                                     nn.BatchNorm2d(90),
#                                     nn.ReLU(),
#                                     nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#                                    )
#
#         self.layer3 = nn.Sequential(
#                                     nn.Conv2d(90, 128, kernel_size=5, stride=1, padding=2),
#                                     nn.BatchNorm2d(128),
#                                     nn.ReLU(),
#
#                                     nn.Conv2d(128, 500, kernel_size=4, stride=1, padding=0),
#                                     nn.BatchNorm2d(500),
#                                     nn.ReLU()
#                                    )
#
#
#         # self.fc1 = nn.Linear(in_features=500, out_features=500, bias=True)
#         self.fc = nn.Linear(in_features=500, out_features=10, bias=True)
#
#
#     def forward(self, x):
#         feat = self.layer1(x)
#         feat_flatten = flatten_images(feat)
#         feat_pca_w1, _ = pca(feat_flatten, 500)
#
#         feat = self.layer2(feat)
#         feat_flatten = flatten_images(feat)
#         feat_pca_w2, _ = pca(feat_flatten, 500)
#
#         feat = self.layer2(feat)
#         out = feat.view(feat.size(0), -1)
#
#         w1, w2, w3 = 0.1101, 0.073, 0.062
#         out = w1 * feat_pca_w1 + w2 * feat_pca_w2 + w3 * out
#         out = self.fc(out)
#
#         return out
