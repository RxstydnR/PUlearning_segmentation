import glob
import os
import random
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from data import data_shuffle, get_data # augmentation
from model import build_model
from PULoss import PULoss, PNULoss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" # Hidden unnecessary logs
os.environ["CUDA_VISIBLE_DEVICES"]=  "4" # Specify the GPU number

parser = argparse.ArgumentParser(
            prog='main.py',
            usage='PU or PNU Learning with deep learning model.',
            description='PU or PNU Learning experiment parameters.'
            )
parser.add_argument('--save_dir', type=str, default="Results", help='path of the save results folder.')
parser.add_argument('--P_dataset', required=True, help='Positive Labeled data path.')
parser.add_argument('--N_dataset', type=str, default=None, help='Negative Labeled data path.')
parser.add_argument('--U_dataset', required=True, help='Positive & Negative Unlabelled data path.')
parser.add_argument('--P_n_sacrifice', type=int, default=0, help='the number of Positive labelled data not used for training.')
parser.add_argument('--N_n_sacrifice', type=int, default=0, help='the number of Negative labelled data not used for training.')
parser.add_argument('--U_n', type=int, default=2000, help='the number of Unlabelled data for training.')
parser.add_argument('--PNU', type=bool, default=False, help='PNU Learning: True')
parser.add_argument('--nnPU', type=bool, default=False, help='nnPU: True, uPU: False')
parser.add_argument('--prior', type=float, default=0.4, help='prior distribution of positive data.')
parser.add_argument('--loss', type=str, default='sigmoid', choices=['sigmoid','softmax'], help='eta in PNU Learning.')
parser.add_argument('--eta', type=float, default=0.0, help='eta in PNU Learning.')
parser.add_argument('--temp', type=float, default=1.0, help='temperature in sigmoid and softmax loss.')
parser.add_argument('--model', type=str, default="ResNet18", choices=['ResNet18','DenseNet121','CNN_paper','CNN'], help='deep learning model.')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--epochs', type=int, default=15, help='training epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate.')
parser.add_argument('--augmentation', type=bool, default=False, help='whether augmentation is applied.')
opt = parser.parse_args()

def main():
    assert (opt.PNU and opt.nnPU) == False, "You can only use either PNULearning or nnPULearning."

    # Data preparation
    if opt.PNU:
        print("============= PNU Learning =============")

        # Positive label data (label: 1)
        P_X = get_data(opt.P_dataset)
        P_Y = np.ones(len(P_X))

        # Negative label data (label: -1)
        N_X = get_data(opt.N_dataset)
        N_Y = np.ones(len(N_X)) * (-1)

        # Unlabel data (label: 0)
        U_X = get_data(opt.U_dataset)
        U_Y = np.zeros(len(U_X))

        # Data shuffle
        P_X, P_Y = data_shuffle(P_X, P_Y)
        N_X, N_Y = data_shuffle(N_X, N_Y)
        U_X, U_Y = data_shuffle(U_X, U_Y)

        # Make dataset
        assert opt.P_n_sacrifice <= len(P_X), "Cannot sacrifice too much P data."
        assert opt.N_n_sacrifice <= len(N_X), "Cannot sacrifice too much N data."
        assert opt.U_n <= len(U_X), "Cannot set bigger U_n than length of Unlabeled data."
        X_train = np.vstack((P_X[:len(P_X)-opt.P_n_sacrifice], N_X[:len(N_X)-opt.N_n_sacrifice], U_X[:opt.U_n]))
        Y_train = np.hstack((P_Y[:len(P_X)-opt.P_n_sacrifice], N_Y[:len(N_X)-opt.N_n_sacrifice], U_Y[:opt.U_n]))
    
    else:
        print("============= PU Learning =============")

        # Positive label data (label: 1)
        P_X = get_data(opt.P_dataset)
        P_Y = np.ones(len(P_X))

        # Unlabel data (label: -1)
        U_X = get_data(opt.U_dataset)
        U_Y = np.ones(len(U_X)) * (-1)

        # Data shuffle
        P_X, P_Y = data_shuffle(P_X, P_Y)
        U_X, U_Y = data_shuffle(U_X, U_Y)

        # Make dataset
        assert opt.P_n_sacrifice <= len(P_X), "Cannot sacrifice too much P data."
        X_train = np.vstack((P_X[:-opt.P_n_sacrifice],U_X[:opt.U_n]))
        Y_train = np.hstack((P_Y[:-opt.P_n_sacrifice],U_Y[:opt.U_n]))


    X_test = np.array(U_X[opt.U_n:]) 
    Y_test = np.array(U_Y[opt.U_n:]) # won't be used.

    # if opt.augmentation:
    #     X_train,Y_train = data_augmentation(X_train,Y_train)
    
    X_train,Y_train = data_shuffle(X_train,Y_train) 
    X_test,Y_test = data_shuffle(X_test,Y_test)

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    print(f"Train: {X_train.shape,Y_train.shape}")
    print(f"Test: {X_test.shape,Y_test.shape}")


    # Model Preparation    
    input_shape = X_train[0].shape
    model = build_model(opt.model, input_shape)

    optimizer = Adam(lr=opt.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    if opt.PNU:
        print("PNU Loss is selected.")
        model.compile(
            loss=PNULoss(prior=opt.prior, eta=opt.eta, loss=opt.loss, temp=opt.temp),
            optimizer=optimizer)
    else:
        if opt.nnPU:
            print("PU Loss is selected.")
        else:
            print("nnPU Loss is selected.")
        model.compile(
            loss=PULoss(prior=opt.prior, nnPU=opt.nnPU, gamma=1, beta=0),
            optimizer=optimizer)
        
    # Training
    model.fit(
        X_train,Y_train, 
        epochs=opt.epochs, 
        batch_size=opt.batchsize,
        verbose=1)
    model.save(os.path.join(result_dir,'model.h5'), include_optimizer=True)

    # Prediction
    pred = model.predict(X_test)
    print(f"The number of test data = {len(X_test)}")
    print(f"Predict Background (positive) = {np.sum(pred>0)}")
    print(f"Predict Aurora     (negative) = {np.sum(pred<0)}")

    # Check prediction probability distribution
    plt.figure()
    sns.displot(pred, bins=20)
    plt.savefig(os.path.join(result_dir,"prediction.png"))
    plt.clf()
    plt.close()
    

if __name__ == "__main__":
    
    result_dir = opt.save_dir
    os.makedirs(result_dir, exist_ok=False)
    
    main()