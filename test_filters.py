import cv2
import numpy as np


# Load the image
img = cv2.imread(r'C:\Users\TUF\Documents\pfee\data_55.jpg')

# Apply median filter with kernel size 5x5
median = cv2.medianBlur(img, 11)

# Save the filtered image
cv2.imwrite('filtered_image.jpg', median)

# Display the original and filtered images side by side
cv2.imshow('Original Image', img)
cv2.imshow('Median Filtered Image', median)
cv2.waitKey(0)
cv2.destroyAllWindows()

#gaussian filter

#
# Choose kernel size and sigma
'''ksize = 10
sigma = 1.0

# Calculate the Gaussian kernel
blurred_image = cv2.GaussianBlur(img, (5, 5), 0)
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

sharpened_image = cv2.filter2D(blurred_image, -1, kernel)

# Save the processed image
cv2.imshow('Original Image', img)
cv2.imshow('gaussian', sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
