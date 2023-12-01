import numpy as np
import cv2 as cv
from PIL import Image
from skimage.exposure import histogram
from skimage.measure import regionprops, label
import yolov5


model = yolov5.load("keremberke/yolov5m-aerial-sheep")  # pretrained model for sheep detection


def get_boxes_yolo(tile):
    """Returns the bounding boxes of a tile using the sheep detector."""
    # x coords refers to columns (width), y coords refers to rows (height)
    tile = (255 - np.array(tile)) / 255.0  # invert image and rescale to 0-1
    tile = (tile * 255).astype(np.uint8)  # rescale to 0-255
    results = model(tile, size=128)  # run inference
    predictions = results.pred[0]  # parse results
    boxes = predictions[:, :4]  # y1, x1, y2, x2
    boxes = [[x[1], x[0], x[3], x[2]] for x in boxes]  # x1, y1, x2, y2
    return np.array(boxes)


def get_boxes_thresh(tile):
    """Returns the bounding boxes of a tile using a thresholding method."""
    boxes = get_region_props(np.array(tile), method="adaptive")
    boxes = [x.bbox for x in boxes]
    return np.array(boxes)


def get_region_props(img, min_area=4000, method="adaptive"):
    '''
    Returns regionprops of connected components in img.
    '''
    # merge the channels
    img = (np.sum(img[:, :, 1:], axis=2) // 2).astype(np.uint8)
    # blur the image
    merged_img = cv.GaussianBlur(img, ksize=(127, 127), sigmaX=0)
    # threshold the image
    if method == "adaptive":
        thresholded_img = 1 - cv.adaptiveThreshold(
            merged_img, 1, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 255, 2
        )
    elif method == "global":
        threshold = get_threshold(merged_img, method="basic")
        thresholded_img = threshold_image(merged_img, threshold, type_="binary")
    else:
        raise ValueError("No such method")
    # label the image using connected components
    labeled_img = label(thresholded_img)
    # get the labels and areas of the connected components
    labels, areas = np.unique(labeled_img, return_counts=True)
    # filter out small regions
    mins = labels[np.where(areas < min_area)]
    for value in mins:
        labeled_img[np.where(labeled_img == value)] = 0
    # return regionprops
    return regionprops(labeled_img)


def get_threshold(image, method="basic"):
    """Find thresholds for each channel based on a basic global thresholding algorithm.

        Args:
            image (numpy.ndarray): input image (H x W x C)
            method (str): Algorithm used for computing threshold. Default is "basic".

    Returns:
            numpy.ndarray: thresholds for each channel (R, G, B)
    """

    if method == "otsu":
        threshold = np.zeros(3)
        for i in range(3):
            threshold[i], _ = cv.threshold(image[:, :, i], 0, 255, type=cv.THRESH_OTSU)
    elif method == "basic":
        threshold = histogram_threshold(image)
    else:
        raise ValueError("No such method.")

    return threshold


def threshold_image(image, threshold, type_="to_one"):
    """Thresholds image based on threshold value and type."""
    if type_ == "to_one":
        return (image <= threshold) * 255 + (image > threshold) * image
    elif type_ == "to_one_inv":
        return (image <= threshold) * image + (image > threshold) * 255
    elif type_ == "binary":
        return (image > threshold) * 255
    elif type_ == "binary_inv":
        return (image <= threshold) * 255
    elif type_ == "to_zero":
        return (image > threshold) * image
    elif type_ == "to_zero_inv":
        return (image <= threshold) * image
    else:
        raise ValueError("No such type")


def histogram_threshold(image):
    '''Obtain threshold for each channel based on histogram analysis.'''
    image = np.array(image).astype(np.uint8)
    # assert a channel axis if not present
    if image.ndim == 2:
        image = image[..., np.newaxis]

    hist, _ = histogram(image, channel_axis=2, source_range="dtype")

    threshold = np.argmax(hist, axis=1)

    for i in range(image.shape[2]):
        condition = False
        while not condition:
            group1_indices = np.arange(threshold[i])
            group2_indices = np.arange(threshold[i], 256)

            mean1 = np.sum(hist[i, group1_indices] * group1_indices) / np.sum(
                hist[i, group1_indices]
            )
            mean2 = np.sum(hist[i, group2_indices] * group2_indices) / np.sum(
                hist[i, group2_indices]
            )

            threshold_new = int(np.mean((mean1, mean2)))

            condition = threshold[i] == threshold_new
            threshold[i] = threshold_new
    return threshold


def compute_iou(boxes):
    # efficient computation of IoU between boxes
    # boxes: (N, 4) ndarray of float
    # result: (N, N) ndarray of overlap between boxes
    boxes = np.array(boxes)
    intersections = compute_intersection(boxes)
    areas = compute_area(boxes)
    
    return intersections / (areas[:, np.newaxis] + areas - intersections)


def compute_intersection(boxes):
    # efficient computation of intersection between boxes
    # boxes: (N, 4) ndarray of float
    # result: (N, N) ndarray of intersection between boxes
    x0 = boxes[:, 0]
    x1 = boxes[:, 2]
    y0 = boxes[:, 1]
    y1 = boxes[:, 3]
    
    x_diff = (np.minimum(x1[:, np.newaxis], x1) - np.maximum(x0[:, np.newaxis], x0)).clip(0)
    y_diff = (np.minimum(y1[:, np.newaxis], y1) - np.maximum(y0[:, np.newaxis], y0)).clip(0)
    
    return x_diff * y_diff


def compute_area(boxes):
    # efficient computation of area of boxes
    # boxes: (N, 4) ndarray of float
    # result: (N,) ndarray of area of boxes
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def get_crop(regionprop, img, size=(512, 512), pad_image=True):
    """Extracts crop from img using coordinates in regionprop

    Args:
        regionprop (tuple or RegionProperties): bounding box coordinates
        img (np.array or PIL.Image): image to crop
        size (tuple, optional): desired crop size. Defaults to (512, 512).
        pad_image (bool, optional): if the output should be padded to desired size. Defaults to True.

    Returns:
        np.array: crop of input img
    """
    # Regionprop is tuple of bbox or RegionProp instance
    try:
        dx, dy, x1, y1 = regionprop.bbox
        regionprop_slice = regionprop.slice
    except AttributeError:
        # regionprop = tuple(map(np.round, regionprop)) 
        regionprop = tuple(map(int, regionprop)) # cast to int
        dx, dy, x1, y1 = regionprop
        regionprop_slice = slice(dx, x1), slice(dy, y1)

    lx, ly = x1 - dx, y1 - dy

    TOP = np.max(((ly - lx) // 2, 0)).astype(int)
    BOTTOM = np.max((np.round((ly - lx) / 2), 0)).astype(int)
    LEFT = np.max(((lx - ly) // 2, 0)).astype(int)
    RIGHT = np.max((np.round((lx - ly) / 2), 0)).astype(int)

    if pad_image:
        try:
            padded_img = np.pad(
                np.array(img)[regionprop_slice],
                ((TOP, BOTTOM), (LEFT, RIGHT), (0, 0)),
                "constant",
                constant_values=255,
            )
        except ValueError:
            padded_img = np.pad(
                np.array(img)[regionprop_slice],
                ((TOP, BOTTOM), (LEFT, RIGHT)),
                "constant",
                constant_values=255,
            )
        image_out = padded_img

    else:

        size = np.round(np.array(size) * np.array((ly, lx)) / np.max((ly, lx))).astype(
            int
        )

        image_out = np.array(img)[regionprop_slice]

    try:
        return np.array(Image.fromarray(image_out).resize(size))
    except ValueError:
        print(f"ValueError with input shape {np.array(img).shape} and processed shape {image_out.shape} and size {size}")
        print(f"regionprop_slice: {regionprop_slice}")
        raise
    