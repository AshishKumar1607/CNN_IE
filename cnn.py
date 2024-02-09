import cv2

import numpy as np

# Parameters
patch_size = 30
omega = 0.95
t_min = 0.15
rho = 0.30

# image enhancement via using cnn
def Cnn_channel(im, size):
   
    b, g, r = cv2.split(im)

    min_channel = cv2.min(cv2.min(r, g), b)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

    dc = cv2.erode(min_channel, kernel)

    return dc


def atmospheric_light(im, dc):
    
    A = np.zeros(3)
    print(A)
    for i in range(3):  # Iterate over R, G, B channels
        A[i] = np.max(im[:, :, i][np.where(dc == np.max(dc))])
    return A


def transmission_map(im, A, size, omega, rho):
  
    t = 1 - omega * Cnn_channel(im / A, size) - rho
    return t


def dehaze(im, t, A, t_min):
  
    J = np.zeros(im.shape)
    for ind in range(3):
        J[:, :, ind] = ((im[:, :, ind] - A[ind]) / np.maximum(t, t_min)) + A[ind]
    return J


image_path = "image_path.jpg"
hazy_image = cv2.imread(image_path, 1)
Cnn_channel_prior = Cnn_channel(hazy_image, patch_size)

A = atmospheric_light(hazy_image, Cnn_channel_prior)
trans_map = transmission_map(hazy_image, A, patch_size, omega, rho)
defogged_image = dehaze(hazy_image, trans_map, A, t_min)

cv2.imshow("defogged", defogged_image)
cv2.waitKey()
cv2.destroyAllWindows()
