import glob
import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import argparse

import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam

from data import data_shuffle, get_data
from model import build_model

from PULoss import PULoss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" # Remove unwanted Logs. (不要なLogを表示させない)
os.environ["CUDA_VISIBLE_DEVICES"]=  "0" # Specify the GPU to be used. (使用するGPUを指定)

parser = argparse.ArgumentParser(
            prog='main.py',
            usage='PU Learning with deep learning model.',
            description='PU Learning experiment parameters.'
            )
parser.add_argument('--P_dataset', required=True, help='Positive Labelled data path.')
parser.add_argument('--U_dataset', required=True, help='Positive & Negative Unlabelled data path.')
parser.add_argument('--model', type=str, default="ResNet18", choices=['ResNet18','DenseNet121','CNN_paper','CNN'], help='deep learning model.')
parser.add_argument('--nnPU', type=bool, default=True, help='nnPU: True, uPU: False')
parser.add_argument('--prior', type=int, default=0.4, help='prior distribution of positive data.')
parser.add_argument('--n_unlabel', type=int, default=2000, help='the number of Unlabelled data for training.')
parser.add_argument('--n_sacrifice', type=int, default=2000, help='the number of Positive labelled data not used for training.')
parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
parser.add_argument('--epochs', type=int, default=30, help='training epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate.')
parser.add_argument('--save_path', type=str, default="Results", help='path of the save results folder.')
opt = parser.parse_args()


def main():

    os.makedirs(opt.save_path, exist_ok=True)

    # Positive labelled data, label value is 1. (Positive（背景）のラベル有りデータ →　全て陽性[+1]とする)
    P_label_X = get_data(opt.P_dataset)
    P_label_Y = np.ones(len(P_label_X))

    # Positive & Negative unlabelled data, label value is 0. (Positive & Negativeのラベル無しデータ →　全て陰性[-1]とする)
    PN_unlabel_X = get_data(opt.U_dataset)
    PN_unlabel_Y = -1*np.ones(len(PN_unlabel_X))

    # Data for training (学習データの作成)
    PN_unlabel_X, PN_unlabel_Y = data_shuffle(PN_unlabel_X, PN_unlabel_Y)
    X_train = np.vstack((P_label_X[opt.n_sacrifice:],PN_unlabel_X[:opt.n_unlabel]))
    Y_train = np.hstack((P_label_Y[opt.n_sacrifice:],PN_unlabel_Y[:opt.n_unlabel]))

    X_test = np.array(PN_unlabel_X[opt.n_unlabel:]) 
    Y_test = np.array(PN_unlabel_Y[opt.n_unlabel:]) # Not used

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    print(f"train={X_train.shape,Y_train.shape}")
    print(f"test={X_test.shape,Y_test.shape}")

    # Data shuffle
    X_train_pu, Y_train_pu = data_shuffle(X_train,Y_train)
    X_test_pu, Y_test_pu = data_shuffle(X_test,Y_test)


    # Model    
    input_shape = X_train_pu[0].shape
    model = build_model(opt.model_name, input_shape)

    optimizer = Adam(lr=opt.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(
        loss = PULoss(prior=opt.prior, nnPU=opt.nnPU),
        optimizer = optimizer,metrics=["accuracy"])

    # Callbacks
    # lr_scheduler = LearningRateScheduler(step_decay)

    # Train
    model.fit(
        X_train_pu, 
        Y_train_pu, 
        epochs=opt.epochs, 
        batch_size=opt.batchsize, 
        # callbacks=[lr_scheduler],
        verbose=1)

    # Save trained model with a number to prevent the model from being overwritten (モデルが上書き保存されないようにモデルに番号を付与して保存する)
    try:
        n_model = len(glob.glob(opt.save_path + "/*.h5")) 
        model.save(os.path.join(opt.save_path,f'model{n_model+1}.h5'), include_optimizer=True)
    except:
        print("Cannot save model")

    # Prediction
    pred = model.predict(X_test_pu)

    print(f"The number of test data = {len(X_test_pu)}")
    print(f"Predict Background (positive) = {np.sum(pred>=0)}")
    print(f"Predict Aurora     (negative) = {np.sum(pred<0)}")


if __name__ == "__main__":
    main()