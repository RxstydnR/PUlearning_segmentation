import copy
import cv2
import numpy as np

from skimage import img_as_ubyte
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


def get_filter_color(pred):    
    """ Get filter color of image pixel corresponding to prediction values. 

    Args:
        pred (float): prediction value

    Returns:
        bin_px (image): binary filter
        filter_px (image): color filter
    """
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Binary Mask
    bin_px = 0 if pred<0 else 255
    
    # Color probability Mask 
    pred_prob = sigmoid(pred) # (matplot colormap accepts value [0,1]or[0,255])

    # Make color filter
    cm = plt.get_cmap("bwr")
    r,g,b,_ = cm(pred_prob*255) # get filter color
    filter_px = np.array([r,g,b],dtype=np.float32)

    return bin_px, filter_px


def patch_segmentation(img, model, patch_size):
    ''' Color filtering for each image based on predictive probability for each pixel.
        
    Args:
        img (image): image data
        model (keras model): PU Learning pre-trained model
        patch_size (int): Size of the patch image
    
    Returns:
        filter_px (image): 
        x_bin_filter (image): binary filter image
        x_color_filter (image): color filter image
        x_superimpose (image): image superimposed on color fiilter.
    '''

    width,height,_ = img.shape
    assert width==height, 'width = height is expected.'
    assert width%patch_size==0, 'patch size is not suitable for width of original image.'
    assert height%patch_size==0, 'patch size is not suitable for height of original image.'
    assert img.dtype=="uint8", 'Only accept uint8 data type image.' 

    def make_margin(x, margin):
        """ add margin to an image
        """
        height,width,_ = x.shape
        new_height = height + margin*2
        new_width = width + margin*2
        new_x = np.zeros((new_height,new_width,3), dtype=np.float32)
        new_x[margin:-margin,margin:-margin] = x
        return new_x
    
    def remove_margin(x, margin):
        """ remove margin from an image
        """
        new_x = x[margin:-margin,margin:-margin]
        return new_x
    
    p = patch_size
    hp = patch_size//2 # = margin
    
    # Preparation of images.
    img = img.astype('float32') / 255.
    x = copy.deepcopy(img)
    x = make_margin(x,hp)

    x_bin_filter = np.zeros((x.shape[0],x.shape[1],1),dtype=np.float32) # binary filter
    x_color_filter = np.zeros(x.shape,dtype=np.float32) # color filter
    x_superimpose = copy.deepcopy(x) # result of segmentation (a filtered image)

    # Make a patch image with a raster scan by 1px.
    width,height,_ = x.shape
    patch_imgs = np.array([x[i:i+p,j:j+p] for i in range(height-p) for j in range(width-p)], dtype=np.float32)
    
    # Prediction.
    preds = model.predict(patch_imgs, batch_size=512)
    v_line = int(patch_imgs.shape[0]**.5)
    preds = preds.reshape((v_line,v_line))
    
    # Filtering by 1px.
    for i in range(hp,height-hp): 
        for j in range(hp,width-hp):
            bin_px, color_px = get_filter_color(preds[i-hp,j-hp])
            x_bin_filter[i,j] = bin_px
            x_color_filter[i,j] = color_px
    
    # Superimpose
    alpha, beta = 0.7, 0.3
    x_superimpose = cv2.addWeighted(x_superimpose, alpha, x_color_filter, beta, 0)

    # Remove margin from each results.
    x = remove_margin(x,hp)
    x_bin_filter = remove_margin(x_bin_filter,hp).astype(np.uint8)
    x_color_filter = remove_margin(x_color_filter,hp)
    x_superimpose = remove_margin(x_superimpose,hp)

    return x_bin_filter, x_color_filter, x_superimpose


def show_segmentation_result(x, x_filter, x_result):
    """ Make a display figure of segmentation result images.

    Args:
        x (image): original image.
        x_filter (image): color filter image.
        x_result (image): original image superinpsed on color filter image.

    Return:
        figure (matplot figure): 1×3 image window figure.
    """

    figure = plt.figure(figsize=(10, 8))
    plt.subplots_adjust(wspace=0.1, hspace=-0.15)
    gs_master = GridSpec(nrows=2, ncols=3)

    gs_1 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0, 0])
    gs_2 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0, 1])
    gs_3 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0, 2])
    gs_4 = GridSpecFromSubplotSpec(nrows=1, ncols=3, subplot_spec=gs_master[1, :])

    axes_1 = figure.add_subplot(gs_1[:, :])
    axes_2 = figure.add_subplot(gs_2[:, :])
    axes_3 = figure.add_subplot(gs_3[:, :])
    axes_4 = figure.add_subplot(gs_4[:, :])

    axes_1.imshow(x)
    axes_1.set_title("Original", fontsize=12)
    axes_1.axis("off")

    axes_2.imshow(x_filter)
    axes_2.set_title("Probability", fontsize=12)
    axes_2.axis("off")

    axes_3.imshow(x_result)
    axes_3.set_title("Segmented", fontsize=12)
    axes_3.axis("off")

    # Color Bar
    cbar = axes_4.figure.colorbar(
                cm.ScalarMappable(cmap='bwr'), ticks=[0, 1],# norm=norm ),
                ax=axes_4, orientation='horizontal', fraction=0.9, pad=0.05, shrink=0.9, aspect=40)#, extend='both')
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_xticklabels(['Negative',  'Positive'])
    cbar.set_label("Probability colorbar",size=14,labelpad=-10)
    axes_4.axis('off')

    return figure


def show_CRF_result(x, x_before, x_after):
    """ Make a display figure of CRF result images.

    Args:
        x (image): original image.
        x_before (image): original binary segmenation image.
        x_after (image): binary segmenation image after appling CRF.

    Return:
        figure (matplot figure): 1×3 image window figure.
    """

    fig, axs = plt.subplots(1, 3, figsize=(10, 6))
    plt.subplots_adjust(wspace=0.1, hspace=-0.15)

    fig.suptitle('Result of CRF', fontsize=16)
    fig.subplots_adjust(top=1.25)

    axs[0].imshow(x)
    axs[0].set_title("Original", fontsize=12)
    axs[0].axis("off")

    x_before = cv2.cvtColor(x_before, cv2.COLOR_GRAY2RGB)
    axs[1].imshow(x_before)
    axs[1].set_title("CRF not applied", fontsize=12)
    axs[1].axis("off")

    axs[2].imshow(x_after)
    axs[2].set_title("CRF applied", fontsize=12)
    axs[2].axis("off")

    return fig