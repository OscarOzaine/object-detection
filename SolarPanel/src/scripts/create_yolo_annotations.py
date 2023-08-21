import os
import cv2
import glob
import torch
import numpy as np
import skimage.measure as km
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from skimage import io
from skimage.filters import threshold_otsu
from torchvision.ops import nms
from concurrent.futures import ThreadPoolExecutor

__all__ = ['create_yolo_annotation']


def load_data(path):
    images = sorted(os.listdir(path))
    def process_image(img):
        image_path = os.path.join(path, img)
        image = cv2.imread(image_path, 0)
        return image

    with ThreadPoolExecutor(max_workers=64) as pool:
        processed_images = pool.map(process_image, images)

    return processed_images

def save_paths(images_path, filename):
    paths = [str(Path(i)) + '\n' for i in sorted(glob.glob(f'{images_path}/*'))]
    with open(filename, "w") as f:
        f.writelines(paths)
        f.write('\n')

# def normalize_bbox(bbox, size=(256, 256)):
#     img_width, img_height = size
#     # Calculate normalized bounding box coordinates
#     x = bbox[1] / img_width
#     y = bbox[0] / img_height
#     w = (bbox[3] - bbox[1]) / img_width
#     h = (bbox[2] - bbox[0]) / img_height

#     # Return the normalized bounding box values
#     return x, y, w, h

def normalize_bbox(bbox, size=(256, 256)):
    img_w, img_h = size
    x = 0.5 * (bbox[1] + bbox[3]) / img_w
    y = 0.5 * (bbox[0] + bbox[2]) / img_h
    w = (bbox[3] - bbox[1]) / img_w
    h = (bbox[2] - bbox[0]) / img_h
    return x, y, w, h

def gen_bboxes(masks_path, normalize=True):
    result = []

    masks = load_data(masks_path)
    for mask in masks:
        bbox = get_bbox(mask, normalize)
        result.append((0, bbox))

    return result

def get_bbox(mask, normalize = True):
    binary = mask > threshold_otsu(mask)
    labeled = km.label(binary)
    props = km.regionprops(labeled)
    
    bboxes = set([p.bbox for p in props])
    if normalize:
        bboxes = list(map(normalize_bbox, bboxes))
    
    return bboxes


def save_labels(img_path, masks_path, labels_path, normalize=True):
    img_files = sorted(os.listdir(img_path))
    masks_files = sorted(os.listdir(masks_path))
    boxes = gen_bboxes(masks_path, normalize)
    if not labels_path.exists():
        os.mkdir(labels_path)

    for image, (cls, box_coords) in zip(img_files, boxes):
        img_name = f'{labels_path}/{Path(image).stem}'
        with open(f'{img_name}.txt', "w") as f:
            for box in box_coords:
                f.write(f'{cls} ')
                for coord in box:
                    f.write(f'{coord} ')
                f.write('\n')


def create_yolo_annotation(images_path='images', masks_path='masks', phase='', filename='phase.txt'): 
    
    labels_path = Path(f'{Path(images_path).parent}/labels/{phase}')

    print(images_path)
    print(masks_path)
    print(labels_path)
    print(phase)
    print()
    
    save_paths(images_path, filename)
    save_labels(images_path, masks_path, labels_path, True)
