import cv2
import numpy as np
import time
import random

# Part 1
image = cv2.imread("bonn.jpeg")

print("Image dimensions: ", image.shape)
print("Image data type: ", image.dtype)

cv2.imshow("bonn", image)
cv2.waitKey(0)

# Part 2
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_image)
cv2.imshow("H", h)
cv2.imshow("S", s)
cv2.imshow("V", v)
cv2.waitKey(0)

# Part 3
start_time = time.time()
brightened_image_loop = np.zeros(image.shape, np.uint8)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        for k in range(image.shape[2]):
            value = int(image[i, j, k]) + 50
            brightened_image_loop[i, j, k] = min(value, 255)
end_time = time.time()
print(f"Time in loop version : {end_time - start_time:.4f}")
cv2.imshow("Brightened using loop", cv2.hconcat([image, brightened_image_loop]))
cv2.waitKey(0)

# Part 4
start_time = time.time()
brightened_image_np = np.clip(image.astype(np.int16) + 50, 0, 255).astype(np.uint8)
end_time = time.time()
print(f"Time in numpy version : {end_time - start_time:.4f}")
cv2.imshow("Brightened using numpy", cv2.hconcat([image, brightened_image_np]))
cv2.waitKey(0)

# Part 5
image_cpy = image.copy()
left_corner = image[0:32, 0:32]
for i in range(3):
    randY = random.randint(0, image.shape[0] - 32)
    randX = random.randint(0, image.shape[1] - 32)
    image_cpy[randY:randY + 32, randX:randX + 32] = left_corner
cv2.imshow("Random copies image", image_cpy)
cv2.waitKey(0)

# Part 6
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
_, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
masked = cv2.bitwise_and(image, image, mask=mask)
combined = cv2.hconcat([image, masked])
cv2.imshow('Original and masked image', combined)
cv2.waitKey(0)

# Part 7
bordered = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))

for i in range(5):
    center_x = random.randint(50, image.shape[1] - 50)
    center_y = random.randint(50, image.shape[0] - 50)
    radius = random.randint(15, 40)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    cv2.circle(bordered, (center_x, center_y), radius, color, 2)

for i in range(5):
    position_x = random.randint(30, image.shape[1] - 100)
    position_y = random.randint(30, image.shape[0] - 30)
    text = f"Label {i+1}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    
    cv2.putText(bordered, text, (position_x, position_y), font, font_scale, color, 2)

# Display the result
cv2.imshow('Bordered Image with Texts and Circles', bordered)
cv2.waitKey(0)
    
