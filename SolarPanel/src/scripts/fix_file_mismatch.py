import os
import glob
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split

from concurrent.futures import ThreadPoolExecutor
from create_yolo_annotations import create_yolo_annotation


def is_image_in_masks(image_filename, masks):
    normalized_mask = []
    for m in masks:
        maskname = m.split(".")[0]
        maskname = maskname.replace("_label", "")
        normalized_mask.append(maskname)

    if image_filename in normalized_mask:
        return True
    return False

def is_mask_in_images(mask_filename, images):
    normalized_images = []
    for m in images:
        maskname = m.split(".")[0]
        normalized_images.append(maskname)

    if mask_filename in normalized_images:
        return True
    return False

def main():
    search_directory = "/home/oscar/Projects/object-detection/SolarPanel/src/data/solar_panels_experiment"
    img_files = sorted(os.listdir(search_directory+"/images/train"))
    masks_files = sorted(os.listdir(search_directory+"/masks/train"))

    files_to_remove = []

    print(len(img_files))
    print(len(masks_files))
    # i = 0
    # for filename in img_files:
    #     name = filename.split(".")[0]
    #     #print(filename, name)
    #     if (not is_image_in_masks(name, masks_files)):
    #         files_to_remove.append(name)

    # for filename in files_to_remove:
    #     file_path = os.path.join(search_directory+"/images/train", filename + ".png")
    #     os.remove(file_path)
    #     print(f"Removed: {filename}")

    files_to_remove = []

    for filename in masks_files:
        original_name = filename.split('.')[0]
        name = original_name.replace("_label", "")
        if (not is_mask_in_images(name, img_files)):
            print('remove', name)
            files_to_remove.append(original_name)

        

    print(files_to_remove)
    print(len(files_to_remove))
    
    for filename in files_to_remove:
        file_path = os.path.join(search_directory+"/masks/train", filename + ".png")
        os.remove(file_path)
        print(f"Removed: {filename}")



if __name__ == '__main__':
    main()
