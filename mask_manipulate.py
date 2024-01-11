from PIL import Image
import numpy as np

import cv2

body_mask = cv2.imread("body_mask.png", cv2.IMREAD_GRAYSCALE)
face_mask = cv2.imread("face_and_hair.png", cv2.IMREAD_GRAYSCALE)

print(body_mask)

final_mask = body_mask - np.bitwise_and(body_mask, face_mask)

cv2.imwrite("final_mask.png", final_mask)
