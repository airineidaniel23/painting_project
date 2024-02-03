import cv2
import numpy as np
import os
import re

orig_refs = []
expected_refs = 7

def process_orig(image_path):
    original_image = cv2.imread(image_path)
    
    height, width = original_image.shape[:2]
    i = 0
    image = original_image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    original_image = cv2.imread(image_path)  # Keep an unmodified copy of the original image for color checking
    image = original_image.copy()  # Work with a copy for processing

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for green color in HSV
    green_lower = np.array([45, 110, 70])  # Adjust these values as needed
    green_upper = np.array([55, 255, 230])

    # Define the range for blue color in HSV
    blue_lower = np.array([60, 110, 70])  # Adjust these values as needed
    blue_upper = np.array([130, 255, 230])

    # Create masks for green and blue
    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)

    # Combine the masks to isolate green and blue colors
    mask_combined = cv2.bitwise_or(mask_green, mask_blue)

    # Apply the combined mask to the original image
    filtered_image = cv2.bitwise_and(image, image, mask=mask_combined)

    # Convert the filtered image to grayscale
    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('origmask.jpg', gray)
    # Apply median filter to reduce image noise
    gray = cv2.medianBlur(gray, 5)  # Using a kernel size of 5

    # Detect edges using Canny
    edged = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Calculate the center of each contour
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            centerX = int(M["m10"] / M["m00"])
            centerY = int(M["m01"] / M["m00"])
            if (centerX + 20 > width
                or centerX - 20 < 0
                or centerY + 20 > height
                or centerY - 20 < 0):
                continue
            # Check if the center pixel in the original image is black
            ul = original_image[centerY - 5, centerX - 5]
            ur = original_image[centerY - 5, centerX + 5]
            bl = original_image[centerY + 5, centerX - 5]
            br = original_image[centerY + 5, centerX + 5]
            if (
                (1.7 * float(ul[0]) > (float(ul[1]) + float(ul[2]))) and (1.7 * float(ur[1]) > (float(ur[0]) + float(ur[2]))) and
                (1.7 * float(br[0]) > (float(br[1]) + float(br[2]))) and (1.7 * float(bl[1]) > (float(bl[0]) + float(bl[2])))
                ):
                #print(f"upper left Center at ({centerX}, {centerY}) is good.")
                i = i + 1
                # Mark the center with a green dot for visualization
                cv2.circle(image, (centerX, centerY), 5, (0, 255, 0), -1)
                orig_refs.append([centerX, centerY])
            else:
                cv2.circle(image, (centerX, centerY), 5, (0, 0, 255), -1)
                #print(f"Center at ({centerX}, {centerY}) is not good.")

    # Save the image with centers marked (if they are black)
    cv2.imwrite("orig_detected.jpg", image)
    print(f"Found {i} centers")

    # Display the result
    #cv2.imshow('Centers Marked on Filtered Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if i == expected_refs:
        return 0
    else:
        return 1

def process_image(image_path, output_path):
    global orig_refs
    original_image = cv2.imread(image_path)
    temp_refs = []
    
    height, width = original_image.shape[:2]
    i = 0
    image = original_image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    original_image = cv2.imread(image_path)  # Keep an unmodified copy of the original image for color checking
    image = original_image.copy()  # Work with a copy for processing

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for green color in HSV
    green_lower = np.array([45, 110, 70])  # Adjust these values as needed
    green_upper = np.array([55, 255, 230])

    # Define the range for blue color in HSV
    blue_lower = np.array([60, 110, 70])  # Adjust these values as needed
    blue_upper = np.array([130, 255, 230])

    # Create masks for green and blue
    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)

    # Combine the masks to isolate green and blue colors
    mask_combined = cv2.bitwise_or(mask_green, mask_blue)

    # Apply the combined mask to the original image
    filtered_image = cv2.bitwise_and(image, image, mask=mask_combined)

    # Convert the filtered image to grayscale
    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('maskgreen.jpg', gray)
    # Apply median filter to reduce image noise
    gray = cv2.medianBlur(gray, 5)  # Using a kernel size of 5

    # Detect edges using Canny
    edged = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Calculate the center of each contour
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            centerX = int(M["m10"] / M["m00"])
            centerY = int(M["m01"] / M["m00"])
            if (centerX + 20 > width
                or centerX - 20 < 0
                or centerY + 20 > height
                or centerY - 20 < 0):
                continue
            # Check if the center pixel in the original image is black
            ul = original_image[centerY - 5, centerX - 5]
            ur = original_image[centerY - 5, centerX + 5]
            bl = original_image[centerY + 5, centerX - 5]
            br = original_image[centerY + 5, centerX + 5]
            if (
                (1.7 * float(ul[0]) > (float(ul[1]) + float(ul[2]))) and (1.7 * float(ur[1]) > (float(ur[0]) + float(ur[2]))) and
                (1.7 * float(br[0]) > (float(br[1]) + float(br[2]))) and (1.7 * float(bl[1]) > (float(bl[0]) + float(bl[2])))
                ):
                #print(f"upper left Center at ({centerX}, {centerY}) is good.")
                # Mark the center with a green dot for visualization
                cv2.circle(image, (centerX, centerY), 5, (0, 255, 0), -1)
                temp_refs.append([centerX, centerY])
                
                i = i + 1
            else:
                cv2.circle(image, (centerX, centerY), 5, (0, 0, 255), -1)
                #print(f"Center at ({centerX}, {centerY}) is not good.")

    print(f"Found {i} centers")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    orig_refs = sorted(orig_refs, key=lambda x: x[0])
    temp_refs = sorted(temp_refs, key=lambda x: x[0])
    if i == expected_refs:
        cv2.imwrite(output_path, image)
        with open(output_path + ".txt", "a") as file:
            for j in range(0,7):
                file.write(f"{temp_refs[j][0]} {temp_refs[j][1]} {orig_refs[j][0]} {orig_refs[j][1]}\n")
    if i == expected_refs:
        return 0
    else:
        return 1

# Directory paths
input_dir = 'vid'
output_dir = 'resultvid'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Regex to match files like 'frame1.jpg', 'frame2.jpg', etc.
pattern = re.compile(r"frame_\d+\.jpg")

# Process each file in the directory

j = 0
k = 0
process_orig("orig.jpg")
for filename in os.listdir(input_dir):
    if pattern.match(filename):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        j = j + process_image(input_path, output_path)
        print(f"Processed {filename}")
        k = k + 1

print(f"We had {j} bad apples out of {k} => {round((float(j)/float(k) * 100),2)}%")