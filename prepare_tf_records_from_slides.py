import os
import sys

sys.path.append(os.getcwd())

import argparse
import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from openslide import OpenSlide, deepzoom
from scampi_preprocessing.utils import compute_iou, get_crop, get_boxes_yolo, get_boxes_thresh


parser = argparse.ArgumentParser()
parser.add_argument("--path_to_slides", type=str, default="./data/NO 6407-6-5/mrxs")
parser.add_argument("--path_to_tfrecords", type=str, default="./data/NO 6407-6-5/tfrecords")
parser.add_argument("--method", type=str, default="yolo")
parser.add_argument("--image_shape", type=tuple, default=(224, 224, 3))
args = parser.parse_args()


if __name__ == "__main__":

    os.makedirs(args.path_to_tfrecords, exist_ok=True)
    
    slides = glob.glob(args.path_to_slides + "/*.mrxs")
    
    
    if args.method == "yolo":
        get_boxes = get_boxes_yolo # use sheep detector
    elif args.method == "thresh":
        get_boxes = get_boxes_thresh # use connected components
    else:
        raise ValueError("method must be either 'yolo' or 'thresh'")
    
    
    for filepath in tqdm(slides, total=len(slides)):
        
        tile_generator = deepzoom.DeepZoomGenerator(
            OpenSlide(filepath), tile_size=2048, overlap=0, limit_bounds=True
        )
        
        num_rows, num_cols = tile_generator.level_tiles[-1]
            
        filename = filepath.split("/")[-1].split(".")[0]
        destination = args.path_to_tfrecords + "/" + filename + ".tfrecords"
        
        with tf.io.TFRecordWriter(destination) as writer:
            for row in tqdm(range(num_rows), total=num_rows):
                for col in tqdm(range(num_cols), total=num_cols):
                    tile = tile_generator.get_tile(tile_generator.level_count - 1, (row, col))
                    boxes = get_boxes(tile)
                    
                    # only keep non-overlapping boxes
                    if len(boxes) > 1: # if there are more than one box
                        try:
                            iou = compute_iou(boxes) * (np.eye(len(boxes)) == 0)
                        except TypeError:
                            print(boxes)
                            raise
                        # only keep non-overlapping boxes
                        boxes = [box for i, box in enumerate(boxes) if np.all(iou[i] < 1e-3)]
                        
                    for box in boxes:
                        
                        crop = get_crop(box, tile, pad_image=False)
                        crop = tf.image.resize(crop, args.image_shape[:2])
                        crop = tf.cast(crop, tf.uint8)
                        crop = tf.io.serialize_tensor(crop).numpy()
                        
                        # create tf example
                        record_bytes = tf.train.Example(
                            features=tf.train.Features(
                                feature={
                                    "image": tf.train.Feature(
                                        bytes_list=tf.train.BytesList(value=[crop])
                                    ),
                                    "slide": tf.train.Feature(
                                        bytes_list=tf.train.BytesList(value=[filepath.encode()])
                                    ),
                                    "row": tf.train.Feature(
                                        int64_list=tf.train.Int64List(value=[row])
                                    ),
                                    "col": tf.train.Feature(
                                        int64_list=tf.train.Int64List(value=[col])
                                    ),
                                    "bbox": tf.train.Feature(
                                        float_list=tf.train.FloatList(value=list(box))
                                    ),
                                }
                            )
                        ).SerializeToString()
                        # write example to file
                        writer.write(record_bytes)
                        