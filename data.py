import os
import cv2
import glob
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_data(PATH):
    """ Read images and make a dataset.

    Args:
        PATH (str): path to image folder.

    Returns:
        X (numpy arr): List of image data.
    """
    X = []
    for img in glob.glob(PATH+"/*"):
        x = np.array(Image.open(img).convert('RGB'))
        X.append(x)
    return np.array(X)

def data_shuffle(X, Y=[]):
    """ Shuffle image data and label, keeping the correspondence.（画像データとラベルデータの対応関係を維持しつつシャッフル）
    """
    np.random.seed(5)
    p = np.random.permutation(len(X))
    if len(Y)>=1: 
        assert len(X)==len(Y), 'length of X and Y is not matched.'
        return X[p], Y[p]
    return X[p]

# def get_data_from_paths(Imgs):
#     """ Read image data and make dataset.（画像データを読み込む.）
#     """
#     X = []
#     for img in Imgs:
#         x = np.array(Image.open(img).convert('RGB'))
#         X.append(x)
#     return np.array(X)

# def get_bw_data_from_paths(Imgs):
#     """ Read image data and make dataset.（画像データを読み込む.）
#     """
#     X = []
#     for img in Imgs:
#         x = np.array(Image.open(img).convert('L'))
#         X.append(x)
#     return np.array(X)




# def data_augmentation(X,Y):
#     assert len(X)==len(Y), 'Length of X and Y must be same.'
    
#     for x in X:
#         P_X_augment.append(x)
#         for _ in range(3):
#             x_ = x.copy()
#             x_ = augmentation(x_, method=opt.method)
#             P_X_augment.append(x_)
#     P_X_augment = np.array(P_X_augment)

#     P_Y = np.ones(len(P_X_augment))
#     PN_Y = PN_unlabel_Y[:n_unlabel]

#     X_train = np.vstack((P_X_augment,PN_X))
#     Y_train = np.hstack((P_Y,PN_Y))
    
#     assert len(X)==len(Y), 'Length of X and Y must be same.'
#     return 


# def augmentation(x, method):

#     def horizontal_flip(image, rate=0.5):
#         if np.random.rand() < rate:
#             image = image[:, ::-1, :]
#         return image

#     def vertical_flip(image, rate=0.5):
#         if np.random.rand() < rate:
#             image = image[::-1, :, :]
#         return image

#     def solt_peppar(image, rate=0.5):
#         if np.random.rand() < rate:
#             row, col, ch = image.shape
#             ps_x = np.random.randint(0, col-1 , 3)
#             ps_y = np.random.randint(0, row-1 , 3)
#             pp_x = np.random.randint(0, col-1 , 3)
#             pp_y = np.random.randint(0, row-1 , 3)
#             image[(ps_y,ps_x)] = (255, 255, 255)
#             image[(pp_y,pp_x)] = (0, 0, 0)
#         return image

#     def median_blur(image, rate=0.5):
#         if np.random.rand() < rate:
#             image = cv2.medianBlur(image,3)
#         return image
    
#     def gamma_correction(image, rate=0.5):
#         if np.random.rand() < rate:
#             gamma = np.random.choice(range(5,10,1))/10
#             table = (np.arange(256) / 255) ** gamma * 255
#             table = np.clip(table, 0, 255).astype(np.uint8)
#             image = cv2.LUT(image, table)
#         return image

#     def adjust(image, rate=0.5):
#         # 明るさ、コントラスト
#         if np.random.rand() < rate:
#             alpha = np.random.choice(range(10,20+1,1))/10
#             beta = np.random.choice(range(0,30+1,5))
#             image = alpha * image + beta
#             image = np.clip(image, 0, 255).astype(np.uint8)
#         return image
    
#     x = horizontal_flip(x)
#     x = vertical_flip(x)
#     x = solt_peppar(x)
#     x = median_blur(x)
#     # x = adjust(x)
#     x = gamma_correction(x)

#     return x

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
