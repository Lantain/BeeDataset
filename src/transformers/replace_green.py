import skimage.exposure
import cv2
import numpy as np

# load image and get dimensions
img = cv2.imread("1.jpg")

# convert to hsv
lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
L = lab[:,:,0]
A = lab[:,:,1]
B = lab[:,:,2]

# negate A
A = (255 - A)

# multiply negated A by B
nAB = 255 * (A/255) * (B/255)
nAB = np.clip((nAB), 0, 255)
nAB = np.uint8(nAB)


# threshold using inRange
range1 = 80
range2 = 240
mask = cv2.inRange(nAB, range1, range2)
mask = 255 - mask

# apply morphology opening to mask
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# antialias mask
mask = cv2.GaussianBlur(mask, (0,0), sigmaX=4, sigmaY=4, borderType = cv2.BORDER_DEFAULT)
mask = skimage.exposure.rescale_intensity(mask, in_range=(127.5,255), out_range=(0,255))

# put white where ever the mask is zero
result = img.copy()
result[mask==0] = (0, 0, 0)

alpha = np.sum(result, axis=-1) > 0
alpha = np.sum(result, axis=-1) > 0
alpha = np.uint8(alpha * 255)
res = np.dstack((result, alpha))
cv2.imwrite('result.png', res)

# write result to disk
cv2.imwrite("result.png", result)

# # display it
# cv2_imshow(nAB)
# cv2_imshow(mask)
# cv2_imshow(result)