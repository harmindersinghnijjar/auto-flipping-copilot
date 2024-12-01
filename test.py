# screenshots\buy_slot_slot1.png
# Get the average color of the image at the upper right corner at a 10x10 pixel area

import cv2
import numpy as np

# Load the image
image = cv2.imread("screenshots/buy_slot_slot2.png")

# Get the average color of the image at the upper right corner at a 10x10 pixel area
average_color = np.mean(image[:10, -10:], axis=(0, 1))

BGR = average_color
# Convert the average color from BGR to RGB
average_color = BGR[::-1]

# Round the average color to integer values
average_color = tuple(map(int, average_color))

# Print the average color as RGB values
print(f"Average color: {average_color}")
