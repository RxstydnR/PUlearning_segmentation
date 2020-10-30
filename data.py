import glob
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import cv2
from tqdm import tqdm
import os

def get_data(PATH):
    """ Read image data and make dataset.（画像データを読み込む.）
    """
    X = []
    for img in tqdm(glob.glob(PATH+"/*.jpg")):
        x = np.array(Image.open(img))
        X.append(x)
    return np.array(X)


def data_shuffle(X, Y=[]):
    """ Shuffle image data and label, keeping the correspondence.（画像データとラベルデータの対応関係を維持しつつシャッフル）
    """
    # np.random.seed(0)
    p = np.random.permutation(len(X))
    if len(Y)>=1: 
        assert len(X)==len(Y), 'length of X and Y is not matched.'
        return X[p], Y[p]
    return X[p]


# def show_train_data(X_train_pu,Y_train_pu, path):
#     X_train_P = X_train_pu[np.where(Y_train_pu==1)]
#     X_train_N = X_train_pu[np.where(Y_train_pu==-1)]

#     n = 50
#     plt.figure(figsize=(20, n//5), dpi=150)
#     for i in range(n):
#         plt.subplot(n//10, n//5, i+1)
#         plt.imshow(X_train_P[i])
#         plt.axis("off")

#     plt.tight_layout(rect=[0,0,1,0.93])
#     plt.suptitle("Background (positive) ",size=30, weight=2)
#     plt.savefig(os.path.join(path,"train_data_positive.pdf"))
#     plt.show()
#     plt.clf()
#     plt.close()

#     plt.figure(figsize=(20, n//5), dpi=150)
#     for i in range(n): 
#         plt.subplot(n//10, n//5, i+1)
#         plt.imshow(X_train_N[i])
#         plt.axis("off")

#     plt.tight_layout(rect=[0,0,1,0.93])
#     plt.suptitle("Aurora (negative) ",size=30, weight=2)
#     plt.savefig(os.path.join(path,"train_data_negative.pdf"))
#     plt.show()
#     plt.clf()
#     plt.close()


# def show_pred_data(X_pred_P,X_pred_N,path):
#     n = 50
    
#     plt.figure(figsize=(20, n//5), dpi=150)
#     for i in range(n):
#         plt.subplot(n//10, n//5, i+1)
#         plt.imshow(X_pred_P[i])
#         plt.axis("off")

#     plt.tight_layout(rect=[0,0,1,0.93])
#     plt.suptitle("Pred to be background (positive) ",size=30, weight=2)
#     plt.savefig(os.path.join(path,"pred_result_positive.pdf"))
#     # plt.show()
#     plt.clf(),plt.close()

#     plt.figure(figsize=(20, n//5), dpi=150)
#     for i in range(n): 
#         plt.subplot(n//10, n//5, i+1)
#         plt.imshow(X_pred_N[i])
#         plt.axis("off")

#     plt.tight_layout(rect=[0,0,1,0.93])
#     plt.suptitle("Pred to be Aurora (negative) ",size=30, weight=2)
#     plt.savefig(os.path.join(path,"pred_result_negative.pdf"))
#     # plt.show()
#     plt.clf(), plt.close()