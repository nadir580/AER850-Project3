import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'C:/Users/PC/Documents/GitHub/motherboard_image.JPEG'

image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_contour = max(contours, key=cv2.contourArea)

mask = np.zeros(gray.shape, np.uint8)
cv2.drawContours(mask, [largest_contour], 0, 255, -1)

result = cv2.bitwise_and(image, image, mask=mask)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Edge Detection (Thresholded Image)')
plt.imshow(thresh, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Mask Image')
plt.imshow(mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Final Extracted PCB')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()

output_path = 'extracted_pcb.jpg'
cv2.imwrite(output_path, result)
print(f"Extracted PCB saved at: {output_path}")