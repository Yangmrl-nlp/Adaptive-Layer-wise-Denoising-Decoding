import numpy as np
import cv2

def add_gaussian_noise(image, mean=0, std=10):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

img = cv2.imread("/Users/yangmorunliu/Desktop/232.jpg")
noisy_img = add_gaussian_noise(img, std=90)
cv2.imwrite("/Users/yangmorunliu/Desktop/gaussian_noise.jpg", noisy_img)
