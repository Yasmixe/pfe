'''import cv2
import numpy as np
# Load the original image and its corresponding mask
original_image = cv2.imread("/home/Yasmine/PycharmProjects/PFEE/augmented_test_benign/crop20_545.jpg")
mask = cv2.imread("/home/Yasmine/PycharmProjects/PFEE/mask_predictions/test_benign/crop20_545_mask.jpg", cv2.IMREAD_GRAYSCALE)
folder_path = "/home/Yasmine/PycharmProjects/PFEE/"
mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
mask = mask.astype(np.uint8)
# Apply the mask to the original image
masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)

output_path = "masked_image-4.png"
blurred_image = cv2.GaussianBlur(masked_image, (7, 7), 0)
#sharpened images
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened_image = cv2.filter2D(blurred_image, -1, kernel)

cv2.imwrite(output_path, sharpened_image)'''


import cv2
import os
import numpy as np

# Path to the folder containing the images



import cv2
import os
import numpy as np

# Path to the image
image_path = r"C:\Users\TUF\Documents\pfee\wissal\resized_image\data_0.jpg"

# Path to the corresponding mask
mask_path = r"C:\Users\TUF\Documents\pfee\wissal\resized_mask\mask_0.jpg"

# Path to the folder where the preprocessed image will be saved
output_folder_path = r"C:\Users\TUF\Documents\pfee\wissal\appl"

# Load the original image
original_image = cv2.imread(image_path)

# Load the corresponding mask
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.resize(mask, (256, 256))
mask = mask.astype(np.uint8)

# Apply the mask to the original image
masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)
blurred_image = cv2.GaussianBlur(masked_image, (7, 7), 0)

# Sharpen the image
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened_image = cv2.filter2D(blurred_image, -1, kernel)

# Save the preprocessed image in the output folder
filename = os.path.basename(image_path)
preprocessed_path = os.path.join(output_folder_path, "preprocessed_" + filename)
cv2.imwrite(preprocessed_path, sharpened_image)


print("Preprocessing completed!")



