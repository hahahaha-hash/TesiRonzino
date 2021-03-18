import numpy as np
from PIL import Image, ImageOps
import cv2 as cv


def create_binary_mask(gray_image_path):
    gray_image = Image.open(gray_image_path)
    gray_image = ImageOps.grayscale(gray_image)
    gray_image = np.array(gray_image)

    w, h  = gray_image.shape

    mask = np.zeros(shape=(w+2, h+2), dtype=np.uint8)

    binary_mask = gray_image.copy()
    i = int(h / 2) # Half of height.
    j = int(w / 2) # Half of width.
    cv.floodFill(binary_mask, mask, (i, j), 255)
    for i in range(w):
        for j in range(h):
            if binary_mask[i][j] != 255:
                binary_mask[i][j] = 0

    binary_mask = cv.GaussianBlur(binary_mask, (25,25), cv.BORDER_DEFAULT)

    return Image.fromarray( binary_mask )


if __name__ == '__main__':
    import sys
    import os

    gray_image_path  = sys.argv[1]
    binary_mark_path = sys.argv[2]

    # Create the binary mask from the gray scale image.
    binary_mask = create_binary_mask(gray_image_path)

    # Save the binary mask on disk.
    outdir_path = os.path.dirname( binary_mark_path )
    os.makedirs(outdir_path, exist_ok=True)
    binary_mask.save(binary_mark_path)
