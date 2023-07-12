import matplotlib.pyplot as plt
import cv2

# Load the image
img = cv2.imread('filtered.jpg')

# Convert the RGB image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Save the grayscale image
cv2.imwrite('gray_image.jpg', gray)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculate the histogram
hist = cv2.calcHist([gray],[0],None,[256],[0,256])

# Plot the histogram
plt.hist(gray.ravel(),256,[0,256])
plt.show()
