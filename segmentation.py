import glob
import os
import random
import cv2
import gc
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from make_filter import patch_segmentation, show_segmentation_result, show_CRF_result
from CRF import apply_crf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" # Hidden unnecessary logs
os.environ["CUDA_VISIBLE_DEVICES"]=  "4" # Specify the GPU number

parser = argparse.ArgumentParser(
            prog='main.py',
            usage='Image segmentation with PU learned model.',
            description=''
            )
parser.add_argument('--dataset', type=str, required=True, help='Image data path.')
parser.add_argument('--patch_size', type=int, default=32, help='Size of patch image for prediction of segmentation values.')
parser.add_argument('--model', type=str, required=True, help='Trained model path.')
parser.add_argument('--save_dir', type=str, default="Results", help='path of the save results folder.')
opt = parser.parse_args()


def main():

    # Load PNULoss model
    model = load_model(opt.model, compile=False)

    # Get image paths
    Imgs_paths = glob.glob(opt.dataset+"/*")

    # Pick up a certain number of images
    # Imgs_paths = random.sample(Imgs_paths,1000) 

    for img_path in tqdm(Imgs_paths):

        # Load image
        x = np.array(Image.open(img_path).convert('RGB'))

        # Segmentation
        bin_filter, color_filter, super_result = patch_segmentation(
            img=x,
            model=model, 
            patch_size=opt.patch_size)

        # CRF
        crf_filter, crf_super_before, crf_super_after = apply_crf(x=x, y=bin_filter)

        # Save
        f_name = os.path.basename(img_path)
        
        plt.imsave(os.path.join(opt.save_dir+"/original",f"{f_name}"), x)
        plt.imsave(os.path.join(opt.save_dir+"/color_filter",f"{f_name}"), color_filter)
        plt.imsave(os.path.join(opt.save_dir+"/color_super",f"{f_name}"), super_result)
        
        cv2.imwrite(os.path.join(opt.save_dir+"/crf_filter_before",f"{f_name}"), bin_filter)#, np.expand_dims(x_bin_filter, axis=-1, cmap="jet")
        cv2.imwrite(os.path.join(opt.save_dir+"/crf_filter_after",f"{f_name}"), crf_filter)
        
        plt.imsave(os.path.join(opt.save_dir+"/crf_result_before",f"{f_name}"), crf_super_before)
        plt.imsave(os.path.join(opt.save_dir+"/crf_result_after",f"{f_name}"), crf_super_after)

        show_segmentation_result(x, color_filter, super_result)
        plt.savefig(os.path.join(opt.save_dir+"/color_filter_display",f"{f_name}"), bbox_inches='tight', pad_inches=0.1)

        show_CRF_result(x,bin_filter,crf_filter)
        plt.savefig(os.path.join(opt.save_dir+"/crf_result_display",f"{f_name}"), bbox_inches='tight', pad_inches=0.1)

        plt.clf()
        plt.close()
        gc.collect()


if __name__ == "__main__":

    os.makedirs(os.path.join(opt.save_dir+"/original"),exist_ok=True)

    os.makedirs(os.path.join(opt.save_dir+"/color_filter"),exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir+"/color_super"),exist_ok=True)
    
    os.makedirs(os.path.join(opt.save_dir+"/crf_filter_before"),exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir+"/crf_filter_after"),exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir+"/crf_result_before"),exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir+"/crf_result_after"),exist_ok=True)

    os.makedirs(os.path.join(opt.save_dir+"/color_filter_display"),exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir+"/crf_result_display"),exist_ok=True)
    
    main()