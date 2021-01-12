import cv2
import numpy as np
from skimage.color import gray2rgb, rgb2gray

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian

""" install: https://pypi.org/project/pydensecrf/

    ```pip install pydensecrf```

"""

def crf(original_image, annotated_image, use_2d=True):
    """ Function which returns the labelled image after applying CRF.

    Args:
        original_image (image): Image which has to labelled
        annotated_image (image): Which has been labelled by some technique(FCN in this case)
        use_2d (bool, optional): CRF for 2D data. Defaults to True.
            if use_2d = True specialised 2D fucntions will be applied
            else Generic functions will be applied

    Returns:
        CRF_result: Image corrected by CRF
    """
    # Converting annotated image to RGB if it is Gray scale
    if(len(annotated_image.shape)<3):
        annotated_image = gray2rgb(annotated_image).astype(np.uint32)
    
    annotated_image = annotated_image.astype(np.uint32)
    
    #Converting the annotations RGB color to single 32 bit integer
    annotated_label = annotated_image[:,:,0].astype(np.uint32) + (annotated_image[:,:,1]<<8).astype(np.uint32) + (annotated_image[:,:,2]<<16).astype(np.uint32)
    
    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)
        
    #Creating a mapping back to 32 bit colors
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16
    
    #Gives no of class labels in the annotated image
    n_labels = len(set(labels.flat)) 
    
    #Setting up the CRF model
    if use_2d :
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.8, zero_unsure=False) # U -> (n_label, flatten shape)
        ''' unary_from_labels method parameters
            gt_prob: float
                The certainty of the ground-truth (must be within (0,1)).

            zero_unsure: bool
                If `True`, treat the label value `0` as meaning "could be anything",
                    i.e. entries with this value will get uniform unary probability.
                If `False`, do not treat the value `0` specially, but just as any other class.
        '''
        d.setUnaryEnergy(U)

        # Smoothness kernel
        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

        # Appearance kernel
        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #Run Inference for 5 steps 
    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP,:]

    CRF_result = MAP.reshape(original_image.shape)

    return CRF_result


def apply_crf(x,y):
    """
    Args:
        x (image): original image
        y (image): segmentation image

    Returns:
        crfimage (image): [description]
        x_overlay (image): [description]
        x_overlay_crf (image): [description]
    """
    
    # GaussianBlur
    # x = cv2.GaussianBlur(x,(5,5),0)

    # Binarization (A binary image loaded has ambiguous pixel values.)
    y[y<=127] = 0
    y[y>127] = 255
    y = cv2.cvtColor(y, cv2.COLOR_GRAY2RGB)

    try:
        crfimage = crf(x,y) # return a RGB image
    except: 
        # if segmentation mask is None.
        crfimage = np.zeros(y.shape, dtype=np.uint8)+255

    x_overlay = cv2.addWeighted(x, 0.7, y, 0.3, 0)
    x_overlay_crf = cv2.addWeighted(x, 0.7, crfimage, 0.3, 0)

    return crfimage, x_overlay, x_overlay_crf
