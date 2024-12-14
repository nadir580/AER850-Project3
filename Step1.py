import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('C:/Users/PC/Documents/GitHub/motherboard_image.JPEG')

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.subplot(2, 3, 2)
plt.title('Grayscale Image')
plt.imshow(gray, cmap='gray')
plt.axis('off')

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
plt.subplot(2, 3, 3)
plt.title('Thresholded Image')
plt.imshow(thresh, cmap='gray')
plt.axis('off')

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_contour = max(contours, key=cv2.contourArea)

mask = np.zeros(gray.shape, np.uint8)
cv2.drawContours(mask, [largest_contour], 0, 255, -1)
plt.subplot(2, 3, 4)
plt.title('Mask')
plt.imshow(mask, cmap='gray')
plt.axis('off')

contour_image = image.copy()
cv2.drawContours(contour_image, [largest_contour], 0, (0, 255, 0), 3)
plt.subplot(2, 3, 5)
plt.title('Contour Detection')
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

result = cv2.bitwise_and(image, image, mask=mask)
plt.subplot(2, 3, 6)
plt.title('Extracted PCB')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()

cv2.imwrite('extracted_pcb.jpg', result)