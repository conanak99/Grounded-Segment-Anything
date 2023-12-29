import cv2

mask_image_path = "download/mask.png"
mask_image_path = "download/mask_dillate.png"

# source: https://stackoverflow.com/questions/74996702/pil-numpy-enlarging-white-areas-of-black-white-mask-image
im = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
res = cv2.morphologyEx(im, cv2.MORPH_DILATE, SE)
cv2.imwrite(mask_image_path, res)
