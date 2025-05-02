import cv2
import os

def preprocess_image_mask(image_path, mask_path, size=(512, 512)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, size)
    image = image / 255.0

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, size)
    mask = (mask > 127).astype("float32")  # binarize

    return image, mask
