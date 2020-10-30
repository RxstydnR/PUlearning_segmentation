import copy
import glob
import os
import random
import cv2
import numpy as np
import argparse

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import load_model

from data import get_data, data_shuffle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def color_filtering(img, pred):    
    """ Make and apply color filters for the predicted values.（予測値に対応させたカラーフィルタを作成し適用する.）

    Args:
        img (arr): image to be predicted
        pred (float): probability value

    Returns:
        filtered_img: color filtered image
        filter_img: color filter
    """
    norm = mpl.colors.Normalize(vmin=-1, vmax=1) # [-1,1] → [0,1]
    cm = plt.get_cmap("bwr")

    pred_norm = norm(pred) # norm(np.tanh(pred))
    r,g,b,_ = cm(pred_norm) # フィルタの色
    
    filter_img = np.zeros((32,32,3),dtype=np.float32)
    filter_img[:] = (r,g,b)
    alpha = 0.3 
    beta = 0.7

    # alpha*img1 + beta*img2 + gamma
    filtered_img = cv2.addWeighted(filter_img,alpha,img,beta,0)
    return filtered_img, filter_img


def save_result(x_out, save_path):
    """ Make result image. （結果画像の作成）

    Args:
        x_out (img): original, filter and filtered image combined image
        save_path (str): save directory path
    """

    fig, ax = plt.subplots(2, 1, dpi=150)
    ax[0].imshow(x_out)
    ax[0].axis("off")

    # フィルター画像
    cbar = ax[1].figure.colorbar(
                mpl.cm.ScalarMappable(cmap='bwr'), # norm=norm ),
                ax=ax[1], orientation='horizontal', fraction=0.9, pad=0.05, shrink=0.9, aspect=40)#, extend='both')
    cbar.set_label("(Aurora) Negative <--> Positive (Background)", size=10)
    cbar.ax.tick_params(labelsize=8)
    ax[1].axis('off')
    
    plt.title("(Left) Original, (Center) Probability, (Right) Segmented.", fontsize=10)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.clf(), plt.close()
    return


def test_patch_image(Imgs, model, patch_size, save_dir):
    ''' Color filtering for each image based on predictive probability for each patch image.
        （各画像に対してパッチ画像ごとに予測確率に基づいたカラーフィルタリング.）

    Args:
        Imgs (array): Multiple images
        model: PU Learning pre-trained model
        patch_size (int): Size of the patch image
        save_dir (str): Save directory path
    '''
    _,width,height,_ = Imgs.shape
    assert width==height, 'width = height is expected.  正方形の画像のみ.'
    assert width%patch_size==0, 'patch size is not suitable for width of original image.'
    assert height%patch_size==0, 'patch size is not suitable for height of original image.'

    p = patch_size
    for n in range(len(Imgs)): 
        x = copy.deepcopy(Imgs[n]) # テスト用画像
        x_filter = np.zeros(Imgs[n].shape,dtype=np.float32) # フィルター用画像
        x_result = copy.deepcopy(Imgs[n]) # 結果表示用（画像のカラーフィルタリング済み）画像
        
        for i in range(height//p): # パッチ画像ループ
            for j in range(width//p):
                patch_img = x[i*p:(i+1)*p,j*p:(j+1)*p]
                patch_img = patch_img[np.newaxis,:,:,:]
                
                pred = model.predict(patch_img) # PU Learning済みモデルで予測
                pred = np.squeeze(pred)
                patch_img = np.squeeze(patch_img)
                filtered_img, filter_img = color_filtering(patch_img,pred)
                x_result[i*p:(i+1)*p,j*p:(j+1)*p] = filtered_img
                x_filter[i*p:(i+1)*p,j*p:(j+1)*p] = filter_img
        
        # 結果表示画像を作成                            
        x_out = cv2.hconcat([x,x_filter,x_result])
        save_path = os.path.join(save_dir,f"{n}.jpg")
        save_result(x_out,save_path)

    return print("=== Done ===")


def test_pixel_image(Imgs, model, patch_size, save_dir):
    ''' Color filtering for each image based on predictive probability for each pixel.
        （各画像に対してピクセルごとに予測確率に基づいたカラーフィルタリング）
    
        Args:
            Imgs (array): Multiple images
            model: PU Learning pre-trained model
            patch_size (int): Size of the patch image
            save_dir (str): Save directory path
    '''
    _,width,height,_ = Imgs.shape
    assert width==height, 'width = height is expected.'
    assert width%patch_size==0, 'patch size is not suitable for width of original image.'
    assert height%patch_size==0, 'patch size is not suitable for height of original image.'
    
    p = patch_size
    hp = patch_size//2 # 中央から両端への長さ
    for n in range(Imgs):
        x = copy.deepcopy(Imgs[n]) # テスト用画像
        x_filter = np.zeros(Imgs[n].shape,dtype=np.float32) # フィルター用画像
        x_result = copy.deepcopy(Imgs[n]) # 結果表示用（画像のカラーフィルタリング済み）画像
        
        # patch画像の作成 (1pxごとにずらしていきながら, パッチ画像を作成する)
        patch_imgs = np.array([x[i-hp:i+hp,j-hp:j+hp] for i in range(hp,height-hp) for j in range(hp,width-hp)])
        preds = model.predict(patch_imgs)
        
        one_side = int(patch_imgs.shape[0]**.5)
        patch_imgs = patch_imgs.reshape((one_side,one_side,p,p,3))
        preds = preds.reshape((one_side,one_side))
                
        for i in tqdm(range(hp,height-hp)): # パッチ画像ループ
            for j in range(hp,width-hp):
                filtered_img, filter_img = color_filtering(patch_imgs[i-hp,j-hp],preds[i-hp,j-hp])
                x_result[i,j] = filtered_img[hp,hp] # 別のどの点でもいいが、中央ピクセルを入れておく
                x_filter[i,j] = filter_img[hp,hp]

        # 結果表示画像を作成                            
        x_out = cv2.hconcat([x,x_filter,x_result])
        save_path = os.path.join(save_dir,f"{n}.jpg")
        save_result(x_out,save_path)

    return print("=== Done ===")



parser = argparse.ArgumentParser(
            prog='segmentation.py',
            usage='Color Segmentation(Filtering) Based on Predictive Probability of pre-PULearned model.',
            description='segmentation parameters.'
            )
parser.add_argument('--dataset', required=True, help='Image to be segmented.')
parser.add_argument('--model', type=str, required=True, help='pre-trained deep learning model.')
parser.add_argument('--patch_size', type=str, default=32, help='patch image size which must be the same size of model input size. ')
parser.add_argument('--save_dir', type=str, default="Results_imgs", help='path of the save results folder.')
opt = parser.parse_args()

def main():

    # Load trained model
    model = load_model(opt.model, compile=False) 

    X = get_data(opt.dataset)
    # X = data_shuffle(X)
    X = X.astype(np.float32)/255.

    os.makedirs(opt.save_dir,exist_ok=False)
    
    # Segmentation
    test_pixel_image(X, model, opt.patch_size, opt.save_dir)


if __name__ == "__main__":
    main()
