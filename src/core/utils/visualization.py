from typing import Optional
import random

import cv2
from matplotlib import pyplot as plt
import numpy as np

# https://github.com/csgerf/kaggle-HuBMAP/blob/master/notebooks/data_utils/data_util.py
BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize(image: np.ndarray, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(image)


def visualize_16bit(image, figsize=(10, 10)):
    # Divide all values by 65535 so we can display the image using matplotlib
    image = image / 65535
    visualize(image, figsize=figsize)


def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2, **kwargs):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return img


def visualize_titles(img, bbox, title, color=BOX_COLOR, thickness=2, font_thickness = 2, font_scale=0.35, **kwargs):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    ((text_width, text_height), _) = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, title, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR,
                font_thickness, lineType=cv2.LINE_AA)
    return img


def visualize_with_mask(image: np.ndarray, 
                        mask: np.ndarray, 
                        original_image: Optional[np.ndarray] = None, 
                        original_mask: Optional[np.ndarray] = None,
                        fontsize: int = 18,
                        figsize: tuple = (10, 10)):
    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=figsize)

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=figsize)

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)


def plot_two_images(img1, img2, figsize=(16,6)):
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.axis('off')


def visualize_multiple( nrows: int, ncols: int, img, transform, fig_width:int=15, fig_height:int = 15):
    fig, axes = plt.subplots(nrows,ncols)
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)
    plt.tight_layout()
    num_iter = 0
    for row in range(nrows):
        for col in range(ncols):
            augmented_img = transform[num_iter](image=img)['image']
            axes[row,col].imshow(augmented_img)
            axes[row,col].grid(False)
            axes[row,col].set_xticks([])
            axes[row,col].set_yticks([])
            num_iter += 1
    return fig, axes
