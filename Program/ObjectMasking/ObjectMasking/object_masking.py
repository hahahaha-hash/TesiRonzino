import create_binary_mask
import crop_image
import make_square_image

import numpy as np
from PIL import Image, ImageOps
import cv2 as cv


def extract_object(photo_path, seg_mask_path):
    # Create the binary mask from the segmentation mask.
    binary_mask = create_binary_mask.create_binary_mask(seg_mask_path)

    # Scale the binary mask to match the input photo.
    photo = Image.open(photo_path)
    binary_mask = binary_mask.resize( photo.size )

    # Multiply the original_image with the binary_mask
    photo_       = np.array( ImageOps.grayscale(photo) )
    binary_mask_ = np.array(binary_mask)
    extracted = cv.bitwise_and(binary_mask_, photo_)
    extracted = cv.bitwise_not(extracted)

    # Convert numpy array to PIL Image.
    extracted = Image.fromarray(extracted)

    # Crop image.
    extracted = crop_image.crop_image(extracted)

    # Make image square.
    min_size = int( max(extracted.size[0], extracted.size[1]) * 1.2 )
    extracted = make_square_image.make_square_image(extracted, min_size, fill_color=(255, 255, 255))

    return extracted


if __name__ == '__main__':
    import sys
    import os

    photo_path    = sys.argv[1]
    seg_mask_path = sys.argv[2]
    output_path   = sys.argv[3]

    extracted = extract_object(photo_path, seg_mask_path)

    # Save the extracted image on disk.
    outdir_path = os.path.dirname( output_path )
    os.makedirs(outdir_path, exist_ok=True)
    extracted.save(output_path)
