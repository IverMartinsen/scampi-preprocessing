import os
import sys

sys.path.append(os.getcwd())

import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.measure import regionprops, label
from scampi_preprocessing.utils import get_crop

parser = argparse.ArgumentParser()
parser.add_argument("--path_to_tile", type=str, default="/Users/ima029/test/SCAMPI/Repository/figures/Example Tiles/tile7.jpg")
args = parser.parse_args()

if __name__ == "__main__":

    tile = np.array(Image.open(args.path_to_tile))
    tile_blurred = cv.GaussianBlur(tile, ksize=(127, 127), sigmaX=0)
    tile_merged = (np.sum(tile_blurred[:, :, 1:], axis=2) // 2).astype(np.uint8)
    tile_thresholded = 1 - cv.adaptiveThreshold(tile_merged, 1, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 255, 2)
    tile_labeled = label(tile_thresholded)
    labels, areas = np.unique(tile_labeled, return_counts=True)
    min_size = 4000
    mins = labels[np.where(areas < min_size)]
    tile_reduced = np.copy(tile_labeled)

    for value in mins:
        tile_reduced[np.where(tile_labeled == value)] = 0

    tile_reduced_boxed = np.repeat(((tile_reduced == 0)*255)[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

    tile_boxed  = np.copy(tile)
    for region in regionprops(tile_reduced):
        x0, y0, x1, y1 = region.bbox
        #cv.rectangle(tile_boxed, (y0, x0), (y1, x1), color=(255, 0, 0), thickness=2)
        cv.rectangle(tile_reduced_boxed, (y0, x0), (y1, x1), color=(255, 0, 0), thickness=3)


    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

    fontsize = 5
    loc = 'left'
    x=.1

    ax1.imshow(tile); ax1.axis('off'); ax1.set_title('1. Original image.\n', fontsize=fontsize, loc=loc, x=x, fontweight='bold')
    ax2.imshow(tile_blurred); ax2.axis('off'); ax2.set_title('2. Image after smoothing.\n', fontsize=fontsize, loc=loc, x=x, fontweight='bold')
    ax3.imshow(tile_merged, cmap='gray'); ax3.axis('off'); ax3.set_title('3. Image after discarding the red channel\n    (summing up the other two).', fontsize=fontsize, loc=loc, x=x, fontweight='bold')
    ax4.imshow((tile_thresholded == 0)*255, cmap='gray'); ax4.axis('off'); ax4.set_title('4. Image after thresholding.\n', fontsize=fontsize, loc=loc, x=x, fontweight='bold')
    ax5.imshow((tile_reduced > 0)*255, cmap='binary'); ax5.axis('off'); ax5.set_title('5. Image after using connected components\n    to filter out small object regions.', fontsize=fontsize, loc=loc, x=x, fontweight='bold')
    ax6.imshow(tile_reduced_boxed); ax6.axis('off'); ax6.set_title('6. Image with bounding boxes based on\n    connected components of remaining regions.', fontsize=fontsize, loc=loc, x=x, fontweight='bold')

    plt.tight_layout()
    plt.savefig('demo.jpg', dpi=1200)
